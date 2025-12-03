import os
import time
import json
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, Request, HTTPException
import httpx

# === NEW: BTC-Guard module ===
from btc_guard.btc_guard import BTCGuard, BTCGuardConfig, BTCDecision

app = FastAPI()

# === ENVIRONMENT VARIABLES ===

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID_AI = os.environ["TELEGRAM_CHAT_ID_AI"]
CHAT_ID_PUMP = os.environ["TELEGRAM_CHAT_ID_PUMP"]
CHAT_ID_DUMP = os.environ["TELEGRAM_CHAT_ID_DUMP"]
CHAT_ID_TRADING = os.environ["TELEGRAM_CHAT_ID_TRADING"]

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "10"))

# === GLOBAL STATE ===

_last_by_key = {}       # cooldown for raw alerts
EPISODES = {}           # per-symbol state

TTL_5M = 20 * 60
TTL_1H = 120 * 60

CORE_SIGNALS = {"A1", "A2"}
CONFIRM_SIGNALS = {"A3", "A4", "A5", "A6", "A9", "A10", "A11", "A12"}
MAXPOWER_SIGNALS = {"A9", "A10", "A11", "A12"}
L_SIGNALS = {"A9L", "A10L", "A11L"}

# === TIMEZONE (MSK) ===

MSK_TZ = timezone(timedelta(hours=3))

# === NEW: BTC-Guard instance ===
btc_guard = BTCGuard(BTCGuardConfig())


def _now() -> float:
    return time.time()


def _now_msk_str() -> str:
    """Дата+время, используется в шапке сообщений."""
    return datetime.now(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S MSK")


def _now_msk_time_str() -> str:
    """Только время, для сигнальных строк."""
    return datetime.now(MSK_TZ).strftime("%H:%M:%S MSK")


# === HELPERS ===

def _fmt_price(p) -> str:
    """Форматируем цену: до 8 знаков, без e-06 и хвостовых нулей."""
    if p is None:
        return "—"
    s = f"{p:.8f}".rstrip("0").rstrip(".")
    return s


async def send_telegram(chat_id: str, text: str, disable_preview: bool = True):
    """
    Отправляем plain text в Telegram.
    При ошибках (особенно 429 Too Many Requests) НЕ роняем приложение.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 429:
            print("[TELEGRAM 429] Too Many Requests. Message dropped.")
            return
        else:
            print(f"[TELEGRAM HTTP ERROR] status={status}, detail={e}")
            return
    except Exception as e:
        print(f"[TELEGRAM EXCEPTION] {e}")
        return


@app.get("/")
async def root():
    return {"status": "ok", "message": "AntiFOMO webhook running"}


# =========================================================
#           LEGACY PUMP/DUMP ENDPOINT  /tv
# =========================================================

@app.post("/tv")
async def tv_webhook(request: Request):
    raw = await request.body()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    if data.get("secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Bad secret")

    symbol = str(data.get("symbol") or "?")
    timeframe = str(data.get("timeframe") or "?")
    side = str(data.get("side") or "?").upper()
    signal_type = str(data.get("signal_type") or "PUMP").upper()

    entry = data.get("entry", "—")
    sl = data.get("sl", "—")
    tps = data.get("tps", [])
    conf = data.get("confidence", "—")
    btc_guard_legacy = data.get("btc_guard", "—")
    volume_fade = data.get("volume_fade", "—")
    liq = data.get("liquidity_mode", "HL")
    reason = data.get("reason", "—")
    ts = data.get("ts", "")

    key = f"{symbol}:{timeframe}:{side}:{signal_type}"
    now_ts = _now()
    if now_ts - _last_by_key.get(key, 0) < COOLDOWN:
        return {"status": "cooldown_skip"}
    _last_by_key[key] = now_ts

    tps_str = ", ".join(map(str, tps)) if tps else "—"

    lines = []
    lines.append(f"AntiFOMO {signal_type} Signal")
    lines.append(f"Пара: {symbol}")
    lines.append(f"TF: {timeframe}   SIDE: {side}   LIQ: {liq}")
    lines.append(f"Entry: {entry}   SL: {sl}")
    lines.append(f"TPs: {tps_str}   Conf: {conf}")
    lines.append(f"BTC-Guard: {btc_guard_legacy}   VolFade: {volume_fade}")
    if ts:
        lines.append(f"Time: {ts}")
    if reason:
        lines.append(f"Reason: {reason}")

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    return {"status": "ok"}


# =========================================================
#            ANTI-FOMO v0.9.x AGGREGATOR
# =========================================================

def _get_episode(symbol: str) -> dict:
    ep = EPISODES.get(symbol)
    if ep is None:
        ep = {
            "state": "IDLE",  # IDLE / SETUP / FLIPPED
            "signals": {},    # name -> {active, ts_str, server_ts, timeframe, price}
            "high_pump": None,
            "high_pump_ts": None,
            "low_after_pump": None,
            "low_after_pump_ts": None,
            "setup_ts": None,
            "last_upgrade_ts": None,
            "last_retest_ts": None,
            "last_strength": 0,
        }
        EPISODES[symbol] = ep
    return ep


def _update_ttl(ep: dict, now_ts: float):
    """Disable expired signals by TTL."""
    for name, info in ep["signals"].items():
        if not info.get("active", False):
            continue
        tf = info.get("timeframe", "")
        age = now_ts - info.get("server_ts", now_ts)
        if tf in ("5", "5m", "5M", "5min"):
            if age > TTL_5M:
                info["active"] = False
        else:
            if age > TTL_1H:
                info["active"] = False


def _set_signal(ep: dict, name: str, timeframe: str, price: float):
    ep["signals"].setdefault(name, {})
    info = ep["signals"][name]
    info["active"] = True
    info["ts_str"] = _now_msk_time_str()  # только время
    info["server_ts"] = _now()
    info["timeframe"] = timeframe
    info["price"] = price


def _signal_active(ep: dict, name: str) -> bool:
    return ep["signals"].get(name, {}).get("active", False)


def _count_confirms(ep: dict) -> int:
    return sum(_signal_active(ep, s) for s in CONFIRM_SIGNALS)


def _has_maxpower(ep: dict) -> bool:
    return any(_signal_active(ep, s) for s in MAXPOWER_SIGNALS)


def _count_l_signals(ep: dict) -> int:
    return sum(_signal_active(ep, s) for s in L_SIGNALS)


def _btc_guard_ok(btc_corr, btc_trend):
    try:
        corr = float(btc_corr)
    except Exception:
        corr = 0.0
    trend = str(btc_trend).upper()

    if trend == "UP" and corr > 0.7:
        return False, f"Корреляция: {corr:.2f}, BTC: UP — BLOCK"
    return True, f"Корреляция: {corr:.2f}, BTC: {trend}"


def _format_signals_block(ep: dict) -> str:
    """Возвращаем список строк: индикатор — статус, цена, время."""
    lines = []

    def add(name, desc):
        info = ep["signals"].get(name, {})
        mark = "✓" if info.get("active", False) else "✗"
        tss = info.get("ts_str", "")
        price = info.get("price", None)
        price_str = _fmt_price(price)
        if tss and price is not None:
            lines.append(f"{name} – {desc}: {mark}, цена {price_str}, время {tss}")
        elif tss:
            lines.append(f"{name} – {desc}: {mark}, время {tss}")
        else:
            lines.append(f"{name} – {desc}: {mark}")

    # Core
    add("A1", "Памп (MicroPump)")
    add("A2", "Перегрев EMA (Overextension)")

    # Confirm
    add("A3", "Volume Spike")
    add("A4", "Rejection Wick")
    add("A5", "LH/LL Structure")
    add("A6", "Momentum Loss")
    add("A9", "SuperTrend 1H DOWN")
    add("A10", "CRSI 20 UP")
    add("A11", "CRSI 80 DOWN")
    add("A12", "Volume Exhaustion 1H")

    # L-сигналы
    add("A9L", "SuperTrend 1H UP (L)")
    add("A10L", "CRSI 80 DOWN (L)")
    add("A11L", "CRSI 20 UP (L)")

    return "\n".join(lines)


def _calc_retest(ep: dict, close_price: float) -> bool:
    """Simple retest: return price back above 50% of pump range."""
    high_pump = ep.get("high_pump")
    low_after = ep.get("low_after_pump")
    if high_pump is None or low_after is None:
        return False
    if high_pump <= low_after:
        return False
    level50 = low_after + (high_pump - low_after) * 0.5
    return close_price is not None and close_price >= level50


# === SENDERS ==============================================================

async def _send_setup(symbol: str, ep: dict, btc_corr, btc_trend, setup_price: float):
    ts = _now_msk_str()
    ep["setup_ts"] = _now()

    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend)
    strength = _count_confirms(ep)
    block = _format_signals_block(ep)

    lines = []
    lines.append("[SETUP READY] ANTI-FOMO SHORT")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время SETUP: {ts}")
    lines.append(f"Цена SETUP: {_fmt_price(setup_price)}")
    lines.append("")
    lines.append("Core (A1, A2):")
    for l in block.split("\n")[:2]:
        lines.append(l)
    lines.append("")
    lines.append(f"Confirm сигналы: {strength}/8")
    for l in block.split("\n")[2:10]:
        lines.append(l)
    lines.append("")
    lines.append(f"BTC-Guard: {btc_comment}")

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    ep["state"] = "SETUP"
    ep["last_strength"] = strength


async def _send_upgrade(symbol: str, ep: dict, old_strength: int, new_strength: int, upgrade_price: float):
    ts = _now_msk_str()
    setup_ts = ep.get("setup_ts")
    setup_str = ""
    if setup_ts:
        setup_str = datetime.fromtimestamp(setup_ts, MSK_TZ).strftime("%Y-%m-%d %H:%M:%S MSK")

    lines = []
    lines.append("[UPGRADE] SHORT SETUP STRENGTHENED")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время UPGRADE: {ts}")
    if setup_str:
        lines.append(f"Время SETUP: {setup_str}")
    lines.append(f"Цена UPGRADE: {_fmt_price(upgrade_price)}")
    lines.append("")
    lines.append(f"Сила сетапа: было {old_strength}/8 -> стало {new_strength}/8")
    lines.append("")
    lines.append("Сигналы:")
    lines.append(_format_signals_block(ep))

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    ep["last_upgrade_ts"] = _now()
    ep["last_strength"] = new_strength


async def _send_retest(symbol: str, ep: dict, close_price: float):
    ts = _now_msk_str()
    hp = ep.get("high_pump")
    lp = ep.get("low_after_pump")

    lines = []
    lines.append("[RETEST] SHORT SCENARIO UNDER ATTACK")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время ретеста: {ts}")
    lines.append("")
    lines.append("Памп:")
    lines.append(f"High пампа: {_fmt_price(hp)}")
    lines.append(f"Low отката: {_fmt_price(lp)}")
    lines.append(f"Цена ретеста: {_fmt_price(close_price)}")
    lines.append("")
    lines.append("Статус: шортовый сценарий под угрозой.")

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))


async def _send_flip(symbol: str, ep: dict, flip_price: float, btc_corr, btc_trend):
    ts = _now_msk_str()
    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend)

    lines = []
    lines.append("[FLIP -> LONG] TREND REVERSAL")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время Flip: {ts}")
    lines.append(f"Цена Flip: {_fmt_price(flip_price)}")
    lines.append("")
    lines.append(f"BTC-Guard: {btc_comment}")
    lines.append("")
    lines.append("L-сигналы:")
    for name in ("A9L", "A10L", "A11L"):
        info = ep["signals"].get(name, {})
        mark = "✓" if info.get("active", False) else "✗"
        tss = info.get("ts_str", "")
        price = info.get("price", None)
        price_str = _fmt_price(price)
        if tss and price is not None:
            lines.append(f"{name}: {mark}, цена {price_str}, время {tss}")
        elif tss:
            lines.append(f"{name}: {mark}, время {tss}")
        else:
            lines.append(f"{name}: {mark}")
    lines.append("")
    lines.append("Статус: шортовый сценарий отменён, контекст сменился на LONG.")

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    ep["state"] = "FLIPPED"


# === MAIN AGGREGATOR =======================================================

async def process_signal_v0(
    symbol: str,
    timeframe: str,
    signal: str,
    side: str,
    ts_str: str,
    close_price: float,
    high_price: float,
    low_price: float,
    btc_corr,
    btc_trend,
):
    now_ts = _now()
    ep = _get_episode(symbol)

    _update_ttl(ep, now_ts)
    _set_signal(ep, signal, timeframe, close_price)

    # Update pump levels from high/low
    if signal in ("A1", "A2"):
        if high_price is not None:
            if ep["high_pump"] is None or high_price > ep["high_pump"]:
                ep["high_pump"] = high_price
                ep["high_pump_ts"] = _now_msk_str()

    if _signal_active(ep, "A1") and _signal_active(ep, "A2"):
        if low_price is not None:
            if ep["low_after_pump"] is None or low_price < ep["low_after_pump"]:
                ep["low_after_pump"] = low_price
                ep["low_after_pump_ts"] = _now_msk_str()

    # Simple RETEST with anti-spam (not more often than every 5 minutes)
    if close_price is not None and _calc_retest(ep, close_price):
        last_rt = ep.get("last_retest_ts")
        if not last_rt or now_ts - last_rt > 300:
            await _send_retest(symbol, ep, close_price)
            ep["last_retest_ts"] = now_ts

    # FLIP logic (>=2 L-signals)
    if _count_l_signals(ep) >= 2 and ep["state"] != "FLIPPED":
        await _send_flip(symbol, ep, close_price, btc_corr, btc_trend)
        return

    # SETUP/UPGRADE logic + телеметрия
    has_core = _signal_active(ep, "A1") and _signal_active(ep, "A2")
    confirms = _count_confirms(ep)
    has_maxpower = _has_maxpower(ep)
    btc_ok, btc_comment = _btc_guard_ok(btc_corr, btc_trend)

    # Телеметрия по сетапу
    print(
        f"[SETUP CHECK] symbol={symbol}, tf={timeframe}, "
        f"core={has_core}, confirms={confirms}, maxpower={has_maxpower}, btc_ok={btc_ok}"
    )

    # Если нет ядра, MaxPower или BTC-Guard блокирует — SETUP не даём
    if not has_core or not has_maxpower or not btc_ok:
        if not btc_ok:
            print(
                f"[BTC-GUARD BLOCK] symbol={symbol}, tf={timeframe}, "
                f"corr={btc_corr}, trend={btc_trend}, comment={btc_comment}"
            )
        ep["last_strength"] = confirms
        return

    # Первый SETUP, когда всё совпало
    if ep["state"] == "IDLE" and confirms >= 3:
        await _send_setup(symbol, ep, btc_corr, btc_trend, close_price)
        return

    # UPGRADE, если сила сетапа выросла
    if ep["state"] == "SETUP" and confirms > ep.get("last_strength", 0):
        old_str = ep.get("last_strength", 0)
        await _send_upgrade(symbol, ep, old_str, confirms, close_price)
        return

    ep["last_strength"] = confirms


# =========================================================
#               NEW /tv_v0 simplified endpoint
#               + BTCGuard integration
# =========================================================

@app.post("/tv_v0")
async def tv_webhook_v0(request: Request):
    raw = await request.body()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    if data.get("secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Bad secret")

    symbol = str(data.get("symbol") or data.get("pair") or "?")
    timeframe = str(data.get("timeframe") or data.get("tf") or "?")
    signal = str(data.get("signal", "?")).upper()
    side = str(data.get("side", "?")).upper()
    ts_str = str(data.get("ts") or data.get("time") or "")

    def _num(x):
        try:
            return float(x)
        except Exception:
            return None

    close_price = _num(data.get("close"))
    high_price = _num(data.get("high"))
    low_price = _num(data.get("low"))

    btc_corr = data.get("btc_corr", 0.0)
    btc_trend = data.get("btc_trend", "UNKNOWN")

    now_ts = _now()

    # 1) BTC tick — только для BTCUSDT, не запускаем AntiFOMO
    if "BTCUSDT" in symbol:
        if close_price is not None:
            btc_guard.on_btc_tick(now_ts=now_ts, close_price=close_price)
        state = btc_guard.get_state().value
        print(f"[BTCGuard] BTC tick symbol={symbol}, price={close_price}, state={state}")
        return {"status": "btc_tick", "symbol": symbol, "btc_state": state}

    # 2) Cooldown на сигналы по альтам
    key = f"v0:{symbol}:{timeframe}:{side}:{signal}"
    if now_ts - _last_by_key.get(key, 0) < COOLDOWN:
        return {"status": "cooldown_skip"}
    _last_by_key[key] = now_ts

    # 3) BTCGuard по монете — решаем, пускать ли сигнал в AntiFOMO
    try:
        corr_val = float(btc_corr)
    except Exception:
        corr_val = 0.0

    decision = btc_guard.handle_signal(
        symbol=symbol,
        side=side,
        btc_corr=corr_val,
        now_ts=now_ts,
        ts_str=ts_str,
        signal_name=signal,
    )

    if decision == BTCDecision.DEFER:
        state = btc_guard.get_state().value
        print(
            f"[BTCGuard] DEFER symbol={symbol}, signal={signal}, "
            f"side={side}, corr={corr_val:.2f}, state={state}"
        )
        return {
            "status": "deferred",
            "symbol": symbol,
            "signal": signal,
            "btc_state": state,
            "corr": corr_val,
        }

    # 4) Если ALLOW — запускаем обычную AntiFOMO-логику
    await process_signal_v0(
        symbol=symbol,
        timeframe=timeframe,
        signal=signal,
        side=side,
        ts_str=ts_str,
        close_price=close_price,
        high_price=high_price,
        low_price=low_price,
        btc_corr=btc_corr,
        btc_trend=btc_trend,
    )

    return {
        "status": "ok",
        "symbol": symbol,
        "signal": signal,
        "btc_state": btc_guard.get_state().value,
    }
