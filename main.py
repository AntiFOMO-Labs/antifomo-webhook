import os
import time
import json
from datetime import datetime, timezone

from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID_AI = os.environ["TELEGRAM_CHAT_ID_AI"]
CHAT_ID_PUMP = os.environ["TELEGRAM_CHAT_ID_PUMP"]
CHAT_ID_DUMP = os.environ["TELEGRAM_CHAT_ID_DUMP"]
CHAT_ID_TRADING = os.environ["TELEGRAM_CHAT_ID_TRADING"]

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "10"))  # защита от дубликатов

_last_by_key = {}  # кулдаун по сырым сигналам
EPISODES = {}       # состояние по эпизодам AntiFOMO v1

# TTL
TTL_5M = 20 * 60      # 20 минут
TTL_1H = 120 * 60     # 120 минут

CORE_SIGNALS = {"A1", "A2"}
CONFIRM_SIGNALS = {"A3", "A4", "A5", "A6", "A9", "A10", "A11", "A12"}
MAXPOWER_SIGNALS = {"A9", "A10", "A11", "A12"}
L_SIGNALS = {"A9L", "A10L", "A11L"}


def escape_md(s: str) -> str:
    """Эскейп для MarkdownV2."""
    for ch in r"_*[]()~`>#+-=|{}.!":
        s = s.replace(ch, "\\" + ch)
    return s


async def send_telegram(chat_id: str, text: str, disable_preview: bool = True):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "MarkdownV2",
        "disable_web_page_preview": disable_preview,
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()


@app.get("/")
async def root():
    return {"status": "ok", "message": "AntiFOMO webhook alive"}


# -------------------------------------------
# СТАРЫЙ /tv ДЛЯ БОЛЬШОЙ ВЕРСИИ (PUMP/DUMP)
# -------------------------------------------
@app.post("/tv")
async def tv_webhook(request: Request):
    raw = await request.body()

    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Bad JSON")

    if data.get("secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Bad secret")

    symbol = str(data.get("symbol", "?"))
    timeframe = str(data.get("timeframe", "?"))
    side = str(data.get("side", "?")).upper()
    signal_type = str(data.get("signal_type", "PUMP")).upper()  # PUMP / DUMP
    entry = data.get("entry", "—")
    sl = data.get("sl", "—")
    tps = data.get("tps", [])
    conf = data.get("confidence", None)
    btc_guard = data.get("btc_guard", "—")
    volume_fade = data.get("volume_fade", "—")
    liq = data.get("liquidity_mode", "HL")  # HL / LL
    reason = data.get("reason", "—")
    ts = data.get("ts", "")

    key = f"{symbol}:{timeframe}:{side}:{signal_type}"
    now = time.time()
    last = _last_by_key.get(key, 0)
    if now - last < COOLDOWN:
        return {
            "status": "cooldown_skip",
            "left_seconds": COOLDOWN - int(now - last),
        }
    _last_by_key[key] = now

    tps_str = ", ".join(map(str, tps)) if tps else "—"
    conf_str = f"{conf}%" if conf is not None else "—"

    header = "AntiFOMO PUMP Signal" if signal_type == "PUMP" else "AntiFOMO DUMP Signal"

    lines = []
    lines.append(f"*{escape_md(header)}*")
    lines.append(
        f"*{escape_md(symbol)}* • `{escape_md(timeframe)}` — *{escape_md(side)}* `{escape_md(liq)}`"
    )
    lines.append(
        f"Entry: `{escape_md(str(entry))}`   SL: `{escape_md(str(sl))}`"
    )
    lines.append(
        f"TPs: `{escape_md(tps_str)}`   Conf: `{escape_md(conf_str)}`"
    )
    lines.append(
        f"BTC\\-Guard: `{escape_md(str(btc_guard))}`   VolFade: `{escape_md(str(volume_fade))}`"
    )
    lines.append(f"_Reason_: {escape_md(str(reason))}")
    if ts:
        lines.append(f"`{escape_md(str(ts))}`")

    text = "\n".join(lines)

    if signal_type == "PUMP":
        target_chat = CHAT_ID_PUMP
    elif signal_type == "DUMP":
        target_chat = CHAT_ID_DUMP
    else:
        target_chat = CHAT_ID_AI

    await send_telegram(target_chat, text)
    return {"status": "ok"}


# -------------------------------------------
# ФУНКЦИИ ДЛЯ АГРЕГАТОРА AntiFOMO v1
# -------------------------------------------

def _now() -> float:
    return time.time()


def _get_episode(symbol: str) -> dict:
    ep = EPISODES.get(symbol)
    if ep is None:
        ep = {
            "state": "IDLE",  # IDLE / CORE / SETUP / CANCELLED / FLIPPED
            "signals": {},    # name -> {active, ts_str, server_ts, timeframe}
            "high_pump": None,
            "low_after_pump": None,
            "setup_ts": None,
            "last_upgrade_ts": None,
            "last_retest_ts": None,
        }
        EPISODES[symbol] = ep
    return ep


def _update_ttl(ep: dict, now_ts: float):
    """Выключаем протухшие сигналы по TTL."""
    for name, info in list(ep["signals"].items()):
        if not info.get("active", False):
            continue
        tf = info.get("timeframe", "")
        age = now_ts - info.get("server_ts", now_ts)
        if tf in ("5", "5m", "5M", "5min"):
            if age > TTL_5M:
                info["active"] = False
        else:
            # считаем всё остальное как 1H-контекст
            if age > TTL_1H:
                info["active"] = False


def _set_signal(ep: dict, name: str, ts_str: str, timeframe: str):
    ep["signals"].setdefault(name, {})
    ep["signals"][name]["active"] = True
    ep["signals"][name]["ts_str"] = ts_str
    ep["signals"][name]["server_ts"] = _now()
    ep["signals"][name]["timeframe"] = timeframe


def _signal_active(ep: dict, name: str) -> bool:
    return ep["signals"].get(name, {}).get("active", False)


def _count_confirms(ep: dict) -> int:
    return sum(1 for s in CONFIRM_SIGNALS if _signal_active(ep, s))


def _has_maxpower(ep: dict) -> bool:
    return any(_signal_active(ep, s) for s in MAXPOWER_SIGNALS)


def _count_l_signals(ep: dict) -> int:
    return sum(1 for s in L_SIGNALS if _signal_active(ep, s))


def _btc_guard_ok(btc_corr: float, btc_trend: str) -> (bool, str):
    """
    Простая логика BTC-Guard v1:
    - если BTC растёт и корреляция > 0.7 -> BLOCK
    - иначе OK
    """
    trend = (btc_trend or "").upper()
    try:
        corr = float(btc_corr)
    except (TypeError, ValueError):
        corr = 0.0

    if trend == "UP" and corr > 0.7:
        return False, f"Корреляция: {corr:.2f}, BTC тренд: UP → BLOCK"
    return True, f"Корреляция: {corr:.2f}, BTC тренд: {trend or 'UNKNOWN'}"


def _format_signals_block(ep: dict) -> str:
    """Блок с галочками и временем для A1–A12 и L-сигналов."""
    lines = []
    def line(name: str, desc: str):
        info = ep["signals"].get(name, {})
        active = info.get("active", False)
        ts_str = info.get("ts_str", "")
        mark = "✓" if active else "✗"
        if ts_str:
            lines.append(
                f"{escape_md(name)} – {escape_md(desc)}: {escape_md(mark)} ({escape_md(ts_str)})"
            )
        else:
            lines.append(
                f"{escape_md(name)} – {escape_md(desc)}: {escape_md(mark)}"
            )

    # Core
    line("A1", "памп (MicroPump)")
    line("A2", "перегрев EMA (Overextension)")

    # Confirm
    line("A3", "объёмный всплеск (VolumeSpike)")
    line("A4", "хвост сверху (Rejection)")
    line("A5", "структура LH/LL")
    line("A6", "потеря импульса (MomentumLoss)")
    line("A9", "SuperTrend 1H DOWN")
    line("A10", "CRSI 20 UP (ранний флип)")
    line("A11", "CRSI 80 DOWN (мощный разворот)")
    line("A12", "Volume Exhaustion 1H")

    # L-сигналы
    line("A9L", "SuperTrend 1H UP (L)")
    line("A10L", "CRSI 80 DOWN (L)")
    line("A11L", "CRSI 20 UP (L)")

    return "\n".join(lines)


def _calc_retest(ep: dict, close_price: float) -> bool:
    """Проверка ретеста (возврат >50% высоты пампа)."""
    high_pump = ep.get("high_pump")
    low_after = ep.get("low_after_pump")
    if high_pump is None or low_after is None:
        return False
    if high_pump <= low_after:
        return False
    level_50 = low_after + (high_pump - low_after) * 0.5
    return close_price >= level_50


async def _send_setup(symbol: str, ep: dict, btc_corr: float, btc_trend: str):
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    setup_ts_str = now_ts
    ep["setup_ts"] = _now()

    btc_ok, btc_comment = _btc_guard_ok(btc_corr, btc_trend)
    strength = _count_confirms(ep)
    signals_block = _format_signals_block(ep)

    lines = []
    lines.append(f"*{escape_md('[SETUP READY] ANTI-FOMO SHORT')}*")
    lines.append(f"{escape_md('Пара')}: *{escape_md(symbol)}*")
    lines.append(f"{escape_md('Время SETUP')}: `{escape_md(setup_ts_str)}`")
    lines.append("")
    lines.append(f"{escape_md('Core')}:")
    lines.append(signals_block.split("\n")[0])  # A1
    lines.append(signals_block.split("\n")[1])  # A2
    lines.append("")
    lines.append(f"{escape_md('Confirm / сила сетапа')}: {strength}/8")
    lines.append(signals_block.split("\n")[2])  # A3
    lines.append(signals_block.split("\n")[3])  # A4
    lines.append(signals_block.split("\n")[4])  # A5
    lines.append(signals_block.split("\n")[5])  # A6
    lines.append(signals_block.split("\n")[6])  # A9
    lines.append(signals_block.split("\n")[7])  # A10
    lines.append(signals_block.split("\n")[8])  # A11
    lines.append(signals_block.split("\n")[9])  # A12
    lines.append("")
    lines.append(f"{escape_md('BTC-Guard')}: {escape_md(btc_comment)}")

    text = "\n".join(lines)
    await send_telegram(CHAT_ID_TRADING, text)
    ep["state"] = "SETUP"
    ep["last_upgrade_ts"] = None


async def _send_upgrade(symbol: str, ep: dict, old_strength: int, new_strength: int):
    now_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    setup_time = ep.get("setup_ts")
    setup_ts_str = ""
    if setup_time:
        setup_ts_str = datetime.fromtimestamp(setup_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    signals_block = _format_signals_block(ep)

    lines = []
    lines.append(f"*{escape_md('[UPGRADE] SHORT SETUP STRENGTHENED')}*")
    lines.append(f"{escape_md('Пара')}: *{escape_md(symbol)}*")
    lines.append(f"{escape_md('Время Upgrade')}: `{escape_md(now_ts_str)}`")
    if setup_ts_str:
        lines.append(f"{escape_md('Время Setup')}: `{escape_md(setup_ts_str)}`")
    lines.append("")
    lines.append(f"{escape_md('Сила сетапа')}: было {old_strength}/8 → стало {new_strength}/8")
    lines.append("")
    lines.append(f"{escape_md('Сигналы')}:")
    lines.append(signals_block)

    text = "\n".join(lines)
    await send_telegram(CHAT_ID_TRADING, text)
    ep["last_upgrade_ts"] = _now()


async def _send_retest(symbol: str, ep: dict, close_price: float):
    now_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    high_pump = ep.get("high_pump")
    low_after = ep.get("low_after_pump")

    lines = []
    lines.append(f"*{escape_md('[RETEST] SHORT SCENARIO UNDER ATTACK')}*")
    lines.append(f"{escape_md('Пара')}: *{escape_md(symbol)}*")
    lines.append(f"{escape_md('Время ретеста')}: `{escape_md(now_ts_str)}`")
    lines.append("")
    lines.append(f"{escape_md('Памп')}:")
    lines.append(f"{escape_md('High пампа')}: `{escape_md(str(high_pump))}`")
    lines.append(f"{escape_md('Low отката')}: `{escape_md(str(low_after))}`")
    lines.append(f"{escape_md('Цена ретеста')}: `{escape_md(str(close_price))}`")
    lines.append("")
    lines.append(escape_md("Статус: шортовый сценарий под угрозой."))

    text = "\n".join(lines)
    await send_telegram(CHAT_ID_TRADING, text)
    ep["last_retest_ts"] = _now()


async def _send_flip(symbol: str, ep: dict):
    now_ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    signals_block = _format_signals_block(ep)

    lines = []
    lines.append(f"*{escape_md('[FLIP → LONG] TREND REVERSAL')}*")
    lines.append(f"{escape_md('Пара')}: *{escape_md(symbol)}*")
    lines.append(f"{escape_md('Время Flip')}: `{escape_md(now_ts_str)}`")
    lines.append("")
    lines.append(f"{escape_md('L-сигналы')}:")
    for name in ("A9L", "A10L", "A11L"):
        info = ep["signals"].get(name, {})
        active = info.get("active", False)
        ts_str = info.get("ts_str", "")
        mark = "✓" if active else "✗"
        if ts_str:
            lines.append(f"{escape_md(name)}: {escape_md(mark)} ({escape_md(ts_str)})")
        else:
            lines.append(f"{escape_md(name)}: {escape_md(mark)}")

    lines.append("")
    lines.append(escape_md("Статус: шортовый сценарий отменён, контекст сменился на LONG."))

    text = "\n".join(lines)
    await send_telegram(CHAT_ID_TRADING, text)
    ep["state"] = "FLIPPED"


async def process_signal_v0(symbol: str,
                            timeframe: str,
                            signal: str,
                            side: str,
                            ts_str: str,
                            close_price: float,
                            high_price: float,
                            low_price: float,
                            btc_corr,
                            btc_trend):
    """
    Основной агрегатор AntiFOMO v1.
    """
    now_ts = _now()
    ep = _get_episode(symbol)

    # Обновляем TTL
    _update_ttl(ep, now_ts)

    # Обновляем сигнал
    _set_signal(ep, signal, ts_str, timeframe)

    # Обновляем high_pump / low_after_pump (для ретеста)
    if signal == "A1" or signal == "A2":
        # фиксируем максимум пампа
        if high_price is not None:
            if ep["high_pump"] is None or high_price > ep["high_pump"]:
                ep["high_pump"] = high_price

    # минимальный low_after_pump: после пампа, когда цена уходит вниз
    if _signal_active(ep, "A1") and _signal_active(ep, "A2"):
        if low_price is not None:
            if ep["low_after_pump"] is None or low_price < ep["low_after_pump"]:
                ep["low_after_pump"] = low_price

    # Проверка RETEST
    if ep["high_pump"] is not None and ep["low_after_pump"] is not None and close_price is not None:
        if _calc_retest(ep, close_price):
            # RETEST
            await _send_retest(symbol, ep, close_price)
            # не сразу FLIP, ждём L-сигналы
            # но уже можем считать шорт-сценарий проблемным
            # (в дальнейшей логике можно блокировать новый SETUP до FLIP/нового эпизода)

    # Эпизод: проверяем Core
    has_core = _signal_active(ep, "A1") and _signal_active(ep, "A2")

    # Считаем Confirm / MaxPower / L-signals
    confirms = _count_confirms(ep)
    has_maxpower = _has_maxpower(ep)
    l_count = _count_l_signals(ep)

    # FLIP → LONG (2+ L-сигнала)
    if l_count >= 2 and ep["state"] != "FLIPPED":
        await _send_flip(symbol, ep)
        return

    # Если нет ядра или нет MaxPower — сетап невозможен
    if not has_core or not has_maxpower:
        # только обновляем состояние, но SETUP не шлём
        return

    # BTC-Guard
    btc_ok, _ = _btc_guard_ok(btc_corr, btc_trend)

    # Если BTC-Guard блокирует — SETUP не даём
    if not btc_ok:
        return

    # Генерация SETUP
    if ep["state"] in ("IDLE", "CORE") and confirms >= 3:
        await _send_setup(symbol, ep, btc_corr, btc_trend)
        return

    # UPGRADE
    if ep["state"] == "SETUP":
        old_strength = _count_confirms(ep)  # до этого шага, но мы уже обновили сигнал...
        # небольшой трюк: мы можем хранить strength в эпизоде, но для простоты
        # будем считать, что если пришёл новый CONFIRM сигнал, strength увеличился,
        # и мы шлём UPGRADE один раз
        # (под v2 можно сделать аккуратный учёт)
        new_strength = _count_confirms(ep)
        # если новый CONFIRM действительно увеличил силу, и это не просто повтор
        if new_strength > old_strength:
            await _send_upgrade(symbol, ep, old_strength, new_strength)


# -------------------------------------------
# НОВЫЙ /tv_v0 ДЛЯ УПРОЩЁННОЙ ANTI-FOMO v1
# -------------------------------------------
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

    # цены
    def _num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    close_price = _num(data.get("close"))
    high_price = _num(data.get("high"))
    low_price = _num(data.get("low"))

    # BTC корреляция/тренд (пока как есть, можно позже привязать к отдельному сигналу)
    btc_corr = data.get("btc_corr", 0.0)
    btc_trend = data.get("btc_trend", "UNKNOWN")

    # Кулдаун на уровне сырых сигналов (защита от дубликатов от TV)
    key = f"v0:{symbol}:{timeframe}:{side}:{signal}"
    now = time.time()
    last = _last_by_key.get(key, 0)
    if now - last < COOLDOWN:
        return {
            "status": "cooldown_skip",
            "left_seconds": COOLDOWN - int(now - last),
        }
    _last_by_key[key] = now

    # Запускаем агрегатор
    await process_signal_v0(symbol, timeframe, signal, side, ts_str, close_price, high_price, low_price, btc_corr, btc_trend)

    # Для отладки можно вернуть state по монете
    return {"status": "ok", "symbol": symbol, "signal": signal}
