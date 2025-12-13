import os
import time
import json
import hmac
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple

from fastapi import FastAPI, Request, HTTPException
import httpx

# === NEW: BTC-Guard module ===
from btc_guard.btc_guard import BTCGuard, BTCGuardConfig, BTCDecision, BTCGuardState
from api_client import BinanceFuturesClient

# === CorrEngine v1 (file-based) ===
# Safe import: if corr_reader.py missing/broken, AntiFOMO will not crash.
try:
    from corr_reader import get_decision as corr_get_decision, get_status as corr_get_status
except Exception:
    corr_get_decision = None
    corr_get_status = None

# -------------------------------------------------
#            LOGGING CONFIG
# -------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("antifomo")

app = FastAPI()


# -------------------------------------------------
#             SETTINGS / ENVIRONMENT
# -------------------------------------------------

class Settings:
    def __init__(self) -> None:
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_ai = os.getenv("TELEGRAM_CHAT_ID_AI")
        self.chat_pump = os.getenv("TELEGRAM_CHAT_ID_PUMP")
        self.chat_dump = os.getenv("TELEGRAM_CHAT_ID_DUMP")
        self.chat_trading = os.getenv("TELEGRAM_CHAT_ID_TRADING")
        self.webhook_secret = os.getenv("WEBHOOK_SECRET")
        self.cooldown = int(os.getenv("COOLDOWN_SECONDS", "10"))

        missing = [
            name for name, value in vars(self).items()
            if value in (None, "") and name not in ("chat_ai", "chat_pump", "chat_dump")
        ]
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


settings = Settings()

BOT_TOKEN = settings.bot_token
CHAT_ID_AI = settings.chat_ai
CHAT_ID_PUMP = settings.chat_pump
CHAT_ID_DUMP = settings.chat_dump
CHAT_ID_TRADING = settings.chat_trading

WEBHOOK_SECRET = settings.webhook_secret
COOLDOWN = settings.cooldown


# -------------------------------------------------
#        SHARED HTTP CLIENT FOR TELEGRAM
# -------------------------------------------------

tg_client: httpx.AsyncClient | None = None
binance_client: BinanceFuturesClient | None = None


# -------------------------------------------------
#                HEARTBEAT CONFIG
# -------------------------------------------------

APP_START_TS = time.time()
HEARTBEAT_INTERVAL_SEC = int(os.getenv("HEARTBEAT_INTERVAL_SEC", "600"))  # 10 минут


# -------------------------------------------------
#               HELPERS (GENERAL)
# -------------------------------------------------

def _now() -> float:
    return time.time()


def _now_msk_str() -> str:
    return datetime.now(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S MSK")


def _now_msk_time_str() -> str:
    return datetime.now(MSK_TZ).strftime("%H:%M:%S MSK")


def _norm_symbol(symbol: str) -> str:
    """Нормализуем символ для Binance/внутренней логики: убираем .P и приводим к UPPER."""
    return (symbol or "").replace(".P", "").strip().upper()


def _fmt_price(p) -> str:
    if p is None:
        return "—"
    try:
        s = f"{p:.8f}".rstrip("0").rstrip(".")
        return s
    except Exception:
        return str(p)


# -------------------------------------------------
#                 TELEGRAM SENDER
# -------------------------------------------------

async def send_telegram(chat_id: str, text: str, disable_preview: bool = True):
    if not BOT_TOKEN:
        logger.error("[TELEGRAM] BOT_TOKEN not set")
        return

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_preview,
    }

    try:
        if tg_client is not None:
            r = await tg_client.post(url, json=payload)
        else:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(url, json=payload)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        logger.warning(f"[TELEGRAM HTTP ERROR] status={status}, detail={e}")
    except Exception as e:
        logger.warning(f"[TELEGRAM EXCEPTION] {e}")


# -------------------------------------------------
#                  HEARTBEAT LOOP
# -------------------------------------------------

HB_COUNTER = {
    "alerts": 0,
    "btc_ticks": 0,
    "allow": 0,
    "defer": 0,
}

async def heartbeat_loop() -> None:
    await asyncio.sleep(10)  # дать сервису подняться

    prev_stats = None

    while True:
        try:
            uptime_min = int((time.time() - APP_START_TS) / 60)
            btc_state = btc_guard.get_state().value

            stats = None
            if binance_client is not None and hasattr(binance_client, "get_orderbook_stats"):
                try:
                    stats = binance_client.get_orderbook_stats()
                except Exception:
                    stats = None

            line1 = f"🟢 AntiFOMO heartbeat | uptime {uptime_min}m | BTCGuard={btc_state}"
            line2 = f"Signals: alerts={HB_COUNTER['alerts']} allow={HB_COUNTER['allow']} defer={HB_COUNTER['defer']} btc_ticks={HB_COUNTER['btc_ticks']}"

            if stats:
                ob_line = (
                    f"OrderBook: ok={stats.get('ok')} cached={stats.get('cached')} stale={stats.get('stale')} "
                    f"blocked={stats.get('blocked')} errors={stats.get('errors')} cache={stats.get('cache_size')}"
                )

                # Дельта за период (не обязательно, но удобно)
                delta_line = ""
                if isinstance(prev_stats, dict):
                    try:
                        delta_line = (
                            f"Δ10m: ok+{stats.get('ok',0)-prev_stats.get('ok',0)} "
                            f"cached+{stats.get('cached',0)-prev_stats.get('cached',0)} "
                            f"stale+{stats.get('stale',0)-prev_stats.get('stale',0)} "
                            f"blocked+{stats.get('blocked',0)-prev_stats.get('blocked',0)}"
                        )
                    except Exception:
                        delta_line = ""

                prev_stats = stats

                msg = "\n".join([line1, line2, ob_line] + ([delta_line] if delta_line else []))
            else:
                msg = "\n".join([line1, line2, "OrderBook: stats unavailable"])

            await send_telegram(CHAT_ID_TRADING, msg)

        except Exception as e:
            logger.warning(f"[HEARTBEAT] error: {e}")

        await asyncio.sleep(HEARTBEAT_INTERVAL_SEC)


# -------------------------------------------------
#                 STARTUP / SHUTDOWN
# -------------------------------------------------

@app.on_event("startup")
async def on_startup():
    global tg_client, binance_client
    logger.info("[STARTUP] Initializing Telegram HTTP client")
    tg_client = httpx.AsyncClient(timeout=15.0)

    try:
        binance_client = BinanceFuturesClient(testnet=False)
        logger.info("[BINANCE] BinanceFuturesClient initialized successfully.")
    except Exception as e:
        logger.warning(f"[BINANCE] Failed to initialize BinanceFuturesClient: {e}")
        binance_client = None

    # старт heartbeat
    asyncio.create_task(heartbeat_loop())
    logger.info("[STARTUP] Heartbeat task scheduled")


@app.on_event("shutdown")
async def on_shutdown():
    global tg_client
    if tg_client is not None:
        logger.info("[SHUTDOWN] Closing Telegram HTTP client")
        await tg_client.aclose()
        tg_client = None


# === GLOBAL STATE ===

_last_by_key: Dict[str, float] = {}   # cooldown for raw alerts
EPISODES: Dict[str, dict] = {}        # per-symbol state

TTL_5M = 20 * 60
TTL_1H = 120 * 60

CORE_SIGNALS = {"A1", "A2"}
CONFIRM_SIGNALS = {"A3", "A4", "A5", "A6", "A9", "A10", "A11", "A12"}
MAXPOWER_SIGNALS = {"A9", "A10", "A11", "A12"}
L_SIGNALS = {"A9L", "A10L", "A11L"}

# === TIMEZONE (MSK) ===
MSK_TZ = timezone(timedelta(hours=3))

# === BTC-Guard INSTANCE + STATE TRACKER ===
btc_guard = BTCGuard(BTCGuardConfig())
BTC_PREV_STATE: BTCGuardState | None = None

# === LIQUIDITY CACHE ===
_liq_cache: Dict[str, Tuple[float, str]] = {}  # symbol_norm -> (ts, level)
LIQ_CACHE_TTL = 5.0  # seconds

# === EPISODE CLEANUP COUNTER ===
_CLEAN_COUNTER = 0
CLEAN_INTERVAL = 500


@app.get("/")
async def root():
    return {"status": "ok", "message": "AntiFOMO webhook running"}


# =========================================================
#           LIQUIDITY ENGINE (BINANCE ORDER BOOK)
# =========================================================

def _get_liquidity_level(symbol: str) -> str:
    """
    Оценка ликвидности по Binance Futures через ордербук.

    Используем get_order_book_with_status() чтобы:
    - чаще отдавать CACHED/STALE вместо ошибок
    - реже забивать логи HTML/403
    """
    symbol_norm = _norm_symbol(symbol)
    now_ts = _now()

    cached = _liq_cache.get(symbol_norm)
    if cached:
        ts, level = cached
        if now_ts - ts < LIQ_CACHE_TTL:
            return level

    if binance_client is None:
        level = "MEDIUM"
        _liq_cache[symbol_norm] = (now_ts, level)
        return level

    try:
        depth, st = binance_client.get_order_book_with_status(symbol=symbol_norm, limit=50)

        if not depth:
            level = "MEDIUM"
            _liq_cache[symbol_norm] = (now_ts, level)
            return level

        bids = depth.get("bids") or []
        asks = depth.get("asks") or []

        def _sum_notional(levels, n=10):
            total = 0.0
            for lvl in levels[:n]:
                try:
                    price = float(lvl[0])
                    qty = float(lvl[1])
                    total += price * qty
                except Exception:
                    continue
            return total

        notional_bids = _sum_notional(bids, n=10)
        notional_asks = _sum_notional(asks, n=10)
        total = notional_bids + notional_asks

        if total >= 20_000_000:
            level = "HIGH"
        elif total >= 5_000_000:
            level = "MEDIUM"
        elif total >= 1_000_000:
            level = "LOW"
        else:
            level = "ULTRA_LOW"

        # Можно логировать статус стакана очень кратко (без спама)
        if st in ("STALE", "NONE"):
            logger.info(f"[LIQUIDITY] symbol={symbol_norm} orderbook_status={st} -> liq={level}")

    except Exception as e:
        logger.warning(f"[LIQUIDITY] ERROR for {symbol_norm}: {e}")
        level = "MEDIUM"

    _liq_cache[symbol_norm] = (now_ts, level)
    return level


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

    secret = str(data.get("secret", ""))
    if not hmac.compare_digest(secret, str(WEBHOOK_SECRET)):
        raise HTTPException(status_code=401, detail="Bad secret")

    symbol_raw = str(data.get("symbol") or "?")
    symbol = _norm_symbol(symbol_raw)

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
            "state": "IDLE",
            "signals": {},
            "high_pump": None,
            "high_pump_ts": None,
            "low_after_pump": None,
            "low_after_pump_ts": None,
            "setup_ts": None,
            "last_upgrade_ts": None,
            "last_retest_ts": None,
            "last_strength": 0,
            "corr_hist": None,
            "corr_hist_samples": 0,
            "last_btc_corr": None,
            "liq_level": None,
        }
        EPISODES[symbol] = ep
    return ep


def _update_ttl(ep: dict, now_ts: float):
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
    info["ts_str"] = _now_msk_time_str()
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


def _update_corr_hist(ep: dict, btc_corr) -> None:
    try:
        corr_now = float(btc_corr)
    except Exception:
        return

    if corr_now < 0 or corr_now > 1.5:
        return

    ep["last_btc_corr"] = corr_now

    prev = ep.get("corr_hist")
    n = ep.get("corr_hist_samples", 0)

    alpha = 0.2
    if prev is None or n == 0:
        new_val = corr_now
        n = 1
    else:
        new_val = prev + alpha * (corr_now - prev)
        n += 1

    ep["corr_hist"] = new_val
    ep["corr_hist_samples"] = n


def _corr_level(c: float) -> str:
    if c < 0.2:
        return "very weak"
    if c < 0.4:
        return "weak"
    if c < 0.7:
        return "medium"
    if c < 0.85:
        return "strong"
    return "extreme"


def _btc_guard_ok(btc_corr, btc_trend, corr_hist=None):
    try:
        corr_now = float(btc_corr)
    except Exception:
        corr_now = 0.0

    trend = (str(btc_trend) or "UNKNOWN").upper()

    if corr_hist is None:
        hist_val = corr_now
    else:
        try:
            hist_val = float(corr_hist)
        except Exception:
            hist_val = corr_now

    level_now = _corr_level(corr_now)
    level_hist = _corr_level(hist_val)

    now_pct = corr_now * 100.0
    hist_pct = hist_val * 100.0

    if trend == "UP" and corr_now > 0.7:
        ok = False
    else:
        ok = True

    comment = (
        f"now {now_pct:.0f}% ({level_now}), "
        f"hist {hist_pct:.0f}% ({level_hist}), "
        f"BTC: {trend}"
    )

    return ok, comment


def _format_liquidity(liq_level: str | None) -> str:
    liq = (liq_level or "UNKNOWN").upper()
    if liq in ("LOW", "ULTRA_LOW"):
        return f"Liquidity: {liq} ⚠"
    if liq in ("HIGH", "MEDIUM"):
        return f"Liquidity: {liq}"
    return "Liquidity: UNKNOWN"


def _format_signals_block(ep: dict) -> str:
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

    add("A1", "Памп (MicroPump)")
    add("A2", "Перегрев EMA (Overextension)")

    add("A3", "Volume Spike")
    add("A4", "Rejection Wick")
    add("A5", "LH/LL Structure")
    add("A6", "Momentum Loss")
    add("A9", "SuperTrend 1H DOWN")
    add("A10", "CRSI 20 UP")
    add("A11", "CRSI 80 DOWN")
    add("A12", "Volume Exhaustion 1H")

    add("A9L", "SuperTrend 1H UP (L)")
    add("A10L", "CRSI 80 DOWN (L)")
    add("A11L", "CRSI 20 UP (L)")

    return "\n".join(lines)


def _calc_retest(ep: dict, close_price: float) -> bool:
    high_pump = ep.get("high_pump")
    low_after = ep.get("low_after_pump")
    if high_pump is None or low_after is None:
        return False
    if high_pump <= low_after:
        return False
    level50 = low_after + (high_pump - low_after) * 0.5
    return close_price is not None and close_price >= level50


def _cleanup_episodes():
    to_del = []
    for symbol, ep in EPISODES.items():
        if ep.get("state") != "IDLE":
            continue
        signals = ep.get("signals", {})
        has_active = any(s.get("active", False) for s in signals.values())
        if not has_active:
            to_del.append(symbol)
    for symbol in to_del:
        EPISODES.pop(symbol, None)
    if to_del:
        logger.info(f"[EPISODES] Cleanup removed {len(to_del)} idle episodes")


# === SENDERS ==============================================================

async def _send_setup(symbol: str, ep: dict, btc_corr, btc_trend, setup_price: float):
    ts = _now_msk_str()
    ep["setup_ts"] = _now()

    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend, ep.get("corr_hist"))
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
    lines.append(_format_liquidity(ep.get("liq_level")))

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    ep["state"] = "SETUP"
    ep["last_strength"] = strength


async def _send_upgrade(symbol: str, ep: dict, old_strength: int, new_strength: int,
                        upgrade_price: float, btc_corr, btc_trend):
    ts = _now_msk_str()
    setup_ts = ep.get("setup_ts")
    setup_str = ""
    if setup_ts:
        setup_str = datetime.fromtimestamp(setup_ts, MSK_TZ).strftime("%Y-%m-%d %H:%M:%S MSK")

    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend, ep.get("corr_hist"))

    lines = []
    lines.append("[UPGRADE] SHORT SETUP STRENGTHENED")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время UPGRADE: {ts}")
    if setup_str:
        lines.append(f"Время SETUP: {setup_str}")
    lines.append(f"Цена UPGRADE: {_fmt_price(upgrade_price)}")
    lines.append("")
    lines.append(f"Сила сетапа: было {old_strength}/8 -> стало {new_strength}/8")
    lines.append(f"BTC-Guard: {btc_comment}")
    lines.append(_format_liquidity(ep.get("liq_level")))
    lines.append("")
    lines.append("Сигналы:")
    lines.append(_format_signals_block(ep))

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
    ep["last_upgrade_ts"] = _now()
    ep["last_strength"] = new_strength


async def _send_retest(symbol: str, ep: dict, close_price: float, btc_corr, btc_trend):
    ts = _now_msk_str()
    hp = ep.get("high_pump")
    lp = ep.get("low_after_pump")
    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend, ep.get("corr_hist"))

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
    lines.append(f"BTC-Guard: {btc_comment}")
    lines.append(_format_liquidity(ep.get("liq_level")))
    lines.append("")
    lines.append("Статус: шортовый сценарий под угрозой.")

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))


async def _send_flip(symbol: str, ep: dict, flip_price: float, btc_corr, btc_trend):
    ts = _now_msk_str()
    _, btc_comment = _btc_guard_ok(btc_corr, btc_trend, ep.get("corr_hist"))

    lines = []
    lines.append("[FLIP -> LONG] TREND REVERSAL")
    lines.append(f"Пара: {symbol}")
    lines.append(f"Время Flip: {ts}")
    lines.append(f"Цена Flip: {_fmt_price(flip_price)}")
    lines.append("")
    lines.append(f"BTC-Guard: {btc_comment}")
    lines.append(_format_liquidity(ep.get("liq_level")))
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
    liq_level: str,
):
    global _CLEAN_COUNTER

    now_ts = _now()
    ep = _get_episode(symbol)

    _CLEAN_COUNTER += 1
    if _CLEAN_COUNTER % CLEAN_INTERVAL == 0:
        _cleanup_episodes()

    _update_ttl(ep, now_ts)
    _update_corr_hist(ep, btc_corr)
    ep["liq_level"] = liq_level

    _set_signal(ep, signal, timeframe, close_price)

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

    if close_price is not None and _calc_retest(ep, close_price):
        last_rt = ep.get("last_retest_ts")
        if not last_rt or now_ts - last_rt > 300:
            await _send_retest(symbol, ep, close_price, btc_corr, btc_trend)
            ep["last_retest_ts"] = now_ts

    if _count_l_signals(ep) >= 2 and ep["state"] != "FLIPPED":
        await _send_flip(symbol, ep, close_price, btc_corr, btc_trend)
        return

    has_core = _signal_active(ep, "A1") and _signal_active(ep, "A2")
    confirms = _count_confirms(ep)
    has_maxpower = _has_maxpower(ep)
    btc_ok, btc_comment = _btc_guard_ok(btc_corr, btc_trend, ep.get("corr_hist"))

    logger.info(
        "[SETUP CHECK] symbol=%s, tf=%s, core=%s, confirms=%s, maxpower=%s, btc_ok=%s, btc_comment=%s, liq=%s",
        symbol, timeframe, has_core, confirms, has_maxpower, btc_ok, btc_comment, liq_level
    )

    if not has_core or not has_maxpower or not btc_ok:
        if not btc_ok:
            logger.info(
                "[BTC-GUARD LOCAL BLOCK] symbol=%s, tf=%s, corr=%s, trend=%s, comment=%s",
                symbol, timeframe, btc_corr, btc_trend, btc_comment
            )
        ep["last_strength"] = confirms
        return

    if ep["state"] == "IDLE" and confirms >= 3:
        await _send_setup(symbol, ep, btc_corr, btc_trend, close_price)
        return

    if ep["state"] == "SETUP" and confirms > ep.get("last_strength", 0):
        old_str = ep.get("last_strength", 0)
        await _send_upgrade(symbol, ep, old_str, confirms, close_price, btc_corr, btc_trend)
        return

    ep["last_strength"] = confirms


# =========================================================
#          SNAPSHOT ПО КОРРЕЛЯЦИЯМ В WARNING/ALERT
# =========================================================

async def _send_btc_corr_snapshot(mode: str):
    header = "BTC-GUARD SNAPSHOT"
    if mode == "WARNING":
        header = "BTC-GUARD WARNING ⚠"
    elif mode == "ALERT":
        header = "BTC-GUARD ALERT 🚨"

    lines = []
    lines.append(header)
    if mode == "WARNING":
        lines.append("BTC ускоряется. Активные шорт-сетапы и их связь с BTC:")
    elif mode == "ALERT":
        lines.append("BTC в сильном импульсе. АнтиФОМО фильтрует монеты по корреляции и ликвидности.")
    else:
        lines.append("Текущая картина шорт-сетапов:")

    lines.append("")
    lines.append("Монета | Corr_now | Corr_hist | Liq | Уровень")

    rows = []

    for symbol, ep in EPISODES.items():
        if ep.get("state") != "SETUP":
            continue

        corr_now = ep.get("last_btc_corr")
        corr_hist = ep.get("corr_hist")
        liq = (ep.get("liq_level") or "UNKNOWN").upper()

        if corr_now is None and corr_hist is None:
            continue

        sort_val = corr_now if corr_now is not None else (corr_hist or 0.0)
        rows.append((symbol, corr_now, corr_hist, liq, sort_val))

    if not rows:
        lines.append("Нет активных монет в состоянии SETUP с данными по корреляции.")
        await send_telegram(CHAT_ID_TRADING, "\n".join(lines))
        return

    rows.sort(key=lambda x: x[4], reverse=True)

    def _fmt_corr(c):
        if c is None:
            return "—"
        return f"{c * 100:.0f}%"

    for symbol, corr_now, corr_hist, liq, _ in rows:
        base = corr_now if corr_now is not None else (corr_hist or 0.0)
        level = _corr_level(base)
        lines.append(
            f"{symbol} | {_fmt_corr(corr_now)} | {_fmt_corr(corr_hist)} | {liq} | {level}"
        )

    await send_telegram(CHAT_ID_TRADING, "\n".join(lines))


# =========================================================
#               NEW /tv_v0 simplified endpoint
#               + BTCGuard integration
# =========================================================

@app.post("/tv_v0")
async def tv_webhook_v0(request: Request):
    global BTC_PREV_STATE
    HB_COUNTER["alerts"] += 1

    raw = await request.body()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    secret = str(data.get("secret", ""))
    if not hmac.compare_digest(secret, str(WEBHOOK_SECRET)):
        raise HTTPException(status_code=401, detail="Bad secret")

    symbol_raw = str(data.get("symbol") or data.get("pair") or "?")
    symbol = _norm_symbol(symbol_raw)

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

    if symbol == "BTCUSDT":
        HB_COUNTER["btc_ticks"] += 1

        prev_state = BTC_PREV_STATE or btc_guard.get_state()
        if close_price is not None:
            btc_guard.on_btc_tick(now_ts=now_ts, close_price=close_price)
        new_state = btc_guard.get_state()

        if prev_state != new_state:
            if new_state == BTCGuardState.WARNING:
                await _send_btc_corr_snapshot(mode="WARNING")
            elif new_state == BTCGuardState.ALERT:
                await _send_btc_corr_snapshot(mode="ALERT")
            elif new_state == BTCGuardState.IDLE:
                await send_telegram(
                    CHAT_ID_TRADING,
                    "BTC-GUARD IDLE ✅\nBTC стабилизировался. AntiFOMO вернулся в обычный режим."
                )

        BTC_PREV_STATE = new_state

        logger.info(
            "[BTCGuard] BTC tick symbol=%s, price=%s, state=%s",
            symbol, close_price, new_state.value
        )
        return {"status": "btc_tick", "symbol": symbol, "btc_state": new_state.value}

    key = f"v0:{symbol}:{timeframe}:{side}:{signal}"
    if now_ts - _last_by_key.get(key, 0) < COOLDOWN:
        return {"status": "cooldown_skip"}
    _last_by_key[key] = now_ts

    # --- CorrEngine v1 gate (file-based) ---
    # CorrEngine пишет /root/antifomo/cache/corr_state.json
    # decision_hint: ALLOW / DEFER / ALERT
    if corr_get_decision is not None and corr_get_status is not None:
        corr_decision = corr_get_decision(symbol, default="DEFER")  # safe default: DEFER
        corr_status = corr_get_status(symbol, default="NO_DATA")

        if corr_decision != "ALLOW":
            HB_COUNTER["defer"] += 1
            logger.info(
                "[CorrEngine] DEFER symbol=%s signal=%s decision=%s status=%s",
                symbol, signal, corr_decision, corr_status
            )
            return {
                "status": "deferred_corr",
                "symbol": symbol,
                "signal": signal,
                "corr_decision": corr_decision,
                "corr_status": corr_status,
            }
    else:
        logger.warning("[CorrEngine] corr_reader not available; skipping corr gate (ALLOW by default)")

    liq_level = _get_liquidity_level(symbol)

    try:
        corr_val = float(btc_corr)
    except Exception:
        corr_val = 0.0

    decision = btc_guard.handle_signal(
        symbol=symbol,
        side=side,
        btc_corr=corr_val,
        liq_level=liq_level,
        now_ts=now_ts,
        ts_str=ts_str,
        signal_name=signal,
    )


    if decision == BTCDecision.DEFER:
        HB_COUNTER["defer"] += 1
        state = btc_guard.get_state().value
        logger.info(
            "[BTCGuard] DEFER symbol=%s, signal=%s, side=%s, corr=%.2f, liq=%s, state=%s",
            symbol, signal, side, corr_val, liq_level, state
        )
        return {
            "status": "deferred",
            "symbol": symbol,
            "signal": signal,
            "btc_state": state,
            "corr": corr_val,
            "liq": liq_level,
        }

    HB_COUNTER["allow"] += 1

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
        liq_level=liq_level,
    )

    return {
        "status": "ok",
        "symbol": symbol,
        "signal": signal,
        "btc_state": btc_guard.get_state().value,
        "liq": liq_level,
    }
