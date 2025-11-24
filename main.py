import os
import time
import json

from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI()

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID_AI = os.environ["TELEGRAM_CHAT_ID_AI"]
CHAT_ID_PUMP = os.environ["TELEGRAM_CHAT_ID_PUMP"]
CHAT_ID_DUMP = os.environ["TELEGRAM_CHAT_ID_DUMP"]
CHAT_ID_TRADING = os.environ["TELEGRAM_CHAT_ID_TRADING"]

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "30"))

_last_by_key = {}


def escape_md(s: str) -> str:
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
# СТАРЫЙ /tv ДЛЯ БОЛЬШОЙ ВЕРСИИ АЛГОРИТМА
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
    signal_type = str(data.get("signal_type", "PUMP")).upper()
    entry = data.get("entry", "—")
    sl = data.get("sl", "—")
    tps = data.get("tps", [])
    conf = data.get("confidence", None)
    btc_guard = data.get("btc_guard", "—")
    volume_fade = data.get("volume_fade", "—")
    liq = data.get("liquidity_mode", "HL")
    reason = data.get("reason", "—")
    ts = data.get("ts", "")

    key = f"{symbol}:{timeframe}:{side}:{signal_type}"
    now = time.time()
    last = _last_by_key.get(key, 0)
    if now - last < COOLDOWN:
        return {"status": "cooldown_skip", "left_seconds": COOLDOWN - int(now - last)}
    _last_by_key[key] = now

    tps_str = ", ".join(map(str, tps)) if tps else "—"
    conf_str = f"{conf}%" if conf is not None else "—"

    header = "AntiFOMO PUMP Signal" if signal_type == "PUMP" else "AntiFOMO DUMP Signal"

    lines = []
    lines.append(f"*{escape_md(header)}*")
    lines.append(
        f"*{escape_md(symbol)}* • `{escape_md(timeframe)}` — *{escape_md(side)}* `{escape_md(liq)}`"
    )
    lines.append(f"Entry: `{escape_md(str(entry))}`   SL: `{escape_md(str(sl))}`")
    lines.append(f"TPs: `{escape_md(tps_str)}`   Conf: `{escape_md(conf_str)}`")
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
# НОВЫЙ ЭНДПОИНТ /tv_v0 ДЛЯ УПРОЩЁННОЙ ANTI-FOMO
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

    symbol = str(data.get("pair") or data.get("symbol") or "?")
    timeframe = str(data.get("tf") or data.get("timeframe") or "?")
    side = str(data.get("side", "?")).upper()
    signal = str(data.get("signal", "?")).upper()
    ts = data.get("time") or data.get("ts") or ""

    key = f"v0:{symbol}:{timeframe}:{side}:{signal}"
    now = time.time()
    last = _last_by_key.get(key, 0)
    if now - last < COOLDOWN:
        return {"status": "cooldown_skip", "left_seconds": COOLDOWN - int(now - last)}

    _last_by_key[key] = now

    lines = []
    lines.append(f"*{escape_md('AntiFOMO v0 signal')}*")
    lines.append(f"*{escape_md(symbol)}* • `{escape_md(timeframe)}` — *{escape_md(side)}*")
    lines.append(f"Signal: `{escape_md(signal)}`")

    if ts:
        lines.append(f"`{escape_md(str(ts))}`")

    text = "\n".join(lines)

    await send_telegram(CHAT_ID_TRADING, text)
    return {"status": "ok"}
