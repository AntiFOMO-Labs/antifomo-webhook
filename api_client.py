#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BINANCE_FAPI_URL = "https://fapi.binance.com"
BINANCE_FAPI_TESTNET_URL = "https://testnet.binancefuture.com"


def _now() -> float:
    return time.monotonic()


def _sleep(sec: float) -> None:
    if sec > 0:
        time.sleep(sec)


def _normalize_symbol(symbol: str) -> str:
    # TradingView may send ".P" suffix – Binance doesn’t use it.
    return symbol.replace(".P", "").strip().upper()


class TokenBucket:
    """Simple token bucket limiter: allows 'rate' tokens per second with 'capacity' burst."""
    def __init__(self, rate: float, capacity: float):
        self.rate = max(0.1, float(rate))
        self.capacity = max(1.0, float(capacity))
        self.tokens = self.capacity
        self.ts = _now()
        self.lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        tokens = max(0.01, float(tokens))
        while True:
            with self.lock:
                now = _now()
                elapsed = now - self.ts
                self.ts = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Need to wait for missing tokens
                missing = tokens - self.tokens
                wait = missing / self.rate

            _sleep(wait)


class BinanceFuturesClient:
    """Обёртка над python-binance для AntiFOMO.
    Variant B: orderbook cache + per-symbol cooldown + global rate limit + retries + stale fallback.
    """

    def __init__(self, testnet: bool = False, recv_window: int = 5000):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET are not set in .env")

        self.testnet = testnet
        self.recv_window = recv_window

        # --- Tuning (env override) ---
        self.ob_cache_ttl = float(os.getenv("ORDERBOOK_CACHE_TTL_SEC", "4.0"))  # cache freshness window
        self.ob_symbol_min_interval = float(os.getenv("ORDERBOOK_SYMBOL_MIN_INTERVAL_SEC", "4.0"))  # per symbol cooldown
        self.ob_global_rps = float(os.getenv("ORDERBOOK_GLOBAL_RPS", "6.0"))  # global limiter
        self.ob_global_burst = float(os.getenv("ORDERBOOK_GLOBAL_BURST", "10.0"))
        self.ob_retries = int(os.getenv("ORDERBOOK_RETRIES", "2"))
        self.ob_backoff_1 = float(os.getenv("ORDERBOOK_BACKOFF_1_SEC", "0.3"))
        self.ob_backoff_2 = float(os.getenv("ORDERBOOK_BACKOFF_2_SEC", "1.0"))

        # --- Client ---
        self.client = Client(api_key, api_secret, testnet=testnet)
        if testnet:
            self.client.FUTURES_URL = BINANCE_FAPI_TESTNET_URL
        else:
            self.client.FUTURES_URL = BINANCE_FAPI_URL

        # --- Limiting + cache ---
        self._ob_bucket = TokenBucket(rate=self.ob_global_rps, capacity=self.ob_global_burst)
        self._ob_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._ob_last_fetch_ts: Dict[str, float] = {}
        self._ob_lock = threading.Lock()

        # --- Stats for heartbeat ---
        self._ob_ok = 0
        self._ob_cached = 0
        self._ob_stale = 0
        self._ob_blocked = 0
        self._ob_errors = 0
        self._ob_last_ok_age_sec = None  # type: Optional[float]

        self.sync_timestamp(initial=True)

    # ---------------- Time sync ----------------

    def sync_timestamp(self, initial: bool = False) -> None:
        try:
            data = self.client.futures_time()
            server_ms = int(data["serverTime"])
            local_ms = int(time.time() * 1000)
            offset = server_ms - local_ms
            self.client.timestamp_offset = offset
            if not initial:
                print(f"[BinanceFuturesClient] Timestamp synced. Offset={offset} ms")
        except Exception as e:
            msg = "WARNING" if initial else "ERROR"
            print(f"[BinanceFuturesClient] {msg}: failed to sync timestamp: {e}")

    def _handle_exception(self, e: Exception, context: str) -> None:
        if isinstance(e, BinanceAPIException):
            print(f"[BinanceFuturesClient] BinanceAPIException in {context}: code={e.code}, msg={e.message}")
            if e.code == -1021:
                print("  → INVALID_TIMESTAMP. Попробую пересинхронизировать время.")
                self.sync_timestamp()
            elif e.code == -2015:
                print("  → Invalid API-key, IP, or permissions. Проверь whitelist IP и права ключа.")
        elif isinstance(e, BinanceRequestException):
            print(f"[BinanceFuturesClient] BinanceRequestException in {context}: {e}")
        else:
            print(f"[BinanceFuturesClient] Unexpected error in {context}: {e}")

    # ---------------- Basic endpoints ----------------

    def ping(self) -> bool:
        try:
            self.client.futures_ping()
            return True
        except Exception as e:
            self._handle_exception(e, "ping")
            return False

    def get_server_time(self) -> Optional[int]:
        try:
            data = self.client.futures_time()
            return int(data["serverTime"])
        except Exception as e:
            self._handle_exception(e, "get_server_time")
            return None

    def get_price(self, symbol: str) -> Optional[float]:
        symbol = _normalize_symbol(symbol)
        try:
            data = self.client.futures_symbol_ticker(symbol=symbol)
            return float(data["price"])
        except Exception as e:
            self._handle_exception(e, f"get_price({symbol})")
            return None

    # ---------------- Order book with cache/limits ----------------

    def get_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        """Backwards-compatible: returns depth dict or None.
        Uses cache, limiter, retries. On temporary blocks returns stale cache if available.
        """
        depth, _status = self.get_order_book_with_status(symbol=symbol, limit=limit)
        return depth

    def get_order_book_with_status(self, symbol: str, limit: int = 50) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        status:
          LIVE   - fetched from Binance now
          CACHED - served from fresh cache
          STALE  - served from stale cache (Binance temporary blocked/error)
          NONE   - no data available
        """
        symbol = _normalize_symbol(symbol)
        limit = max(5, min(int(limit), 500))

        now = _now()

        # 1) serve from fresh cache
        with self._ob_lock:
            cached = self._ob_cache.get(symbol)
            if cached:
                ts, depth = cached
                age = now - ts
                if age <= self.ob_cache_ttl:
                    self._ob_cached += 1
                    self._ob_last_ok_age_sec = age
                    return depth, "CACHED"

        # 2) per-symbol cooldown (avoid burst)
        with self._ob_lock:
            last_fetch = self._ob_last_fetch_ts.get(symbol, 0.0)
            wait_symbol = (last_fetch + self.ob_symbol_min_interval) - now
        if wait_symbol > 0:
            _sleep(wait_symbol)

        # 3) global limiter
        self._ob_bucket.acquire(1.0)

        # 4) fetch with retries
        backoffs = [self.ob_backoff_1, self.ob_backoff_2]
        attempts = 1 + max(0, self.ob_retries)

        last_exc: Optional[Exception] = None
        for i in range(attempts):
            try:
                with self._ob_lock:
                    self._ob_last_fetch_ts[symbol] = _now()

                depth = self.client.futures_order_book(symbol=symbol, limit=limit)

                with self._ob_lock:
                    self._ob_cache[symbol] = (_now(), depth)
                    self._ob_ok += 1
                    self._ob_last_ok_age_sec = 0.0

                return depth, "LIVE"

            except Exception as e:
                last_exc = e
                msg = str(e)
                # Detect HTML/CloudFront blocked symptom inside message
                blocked_like = ("<!DOCTYPE" in msg) or ("Request blocked" in msg) or ("could not be satisfied" in msg) or ("403" in msg)
                if blocked_like:
                    with self._ob_lock:
                        self._ob_blocked += 1
                else:
                    with self._ob_lock:
                        self._ob_errors += 1

                # last attempt? break and fallback
                if i >= attempts - 1:
                    break

                # backoff before retry
                _sleep(backoffs[min(i, len(backoffs) - 1)])

        # 5) fallback to stale cache if exists
        with self._ob_lock:
            cached = self._ob_cache.get(symbol)
            if cached:
                ts, depth = cached
                age = _now() - ts
                self._ob_stale += 1
                self._ob_last_ok_age_sec = age
                return depth, "STALE"

        # 6) no data
        if last_exc is not None:
            self._handle_exception(last_exc, f"get_order_book({symbol})")
        return None, "NONE"

    def get_orderbook_stats(self) -> Dict[str, Any]:
        """For heartbeat / monitoring."""
        with self._ob_lock:
            return {
                "ok": self._ob_ok,
                "cached": self._ob_cached,
                "stale": self._ob_stale,
                "blocked": self._ob_blocked,
                "errors": self._ob_errors,
                "cache_size": len(self._ob_cache),
                "last_ok_age_sec": self._ob_last_ok_age_sec,
                "cfg": {
                    "ttl": self.ob_cache_ttl,
                    "symbol_min_interval": self.ob_symbol_min_interval,
                    "global_rps": self.ob_global_rps,
                    "global_burst": self.ob_global_burst,
                    "retries": self.ob_retries,
                },
            }

    # ---------------- Other endpoints ----------------

    def get_liquidations(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[list]:
        params: Dict[str, Any] = {"limit": max(1, min(limit, 1000))}
        if symbol:
            params["symbol"] = _normalize_symbol(symbol)
        if start_time:
            params["startTime"] = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)

        try:
            data = self.client.futures_liquidation_orders(**params)
            return data
        except Exception as e:
            self._handle_exception(e, "get_liquidations")
            return None

    def http_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[dict]:
        base_url = BINANCE_FAPI_TESTNET_URL if self.testnet else BINANCE_FAPI_URL
        url = base_url + path
        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[BinanceFuturesClient] HTTP error for {url}: {e}")
            return None


if __name__ == "__main__":
    client = BinanceFuturesClient(testnet=False)
    print("Ping:", client.ping())
    print("Server time:", client.get_server_time())
    print("BTCUSDT price:", client.get_price("BTCUSDT"))

    # Quick stress test (won't burst Binance due to caching/limiter)
    for _ in range(10):
        depth, st = client.get_order_book_with_status("BTCUSDT", limit=50)
        print("Orderbook status:", st, "| bids:", len(depth.get("bids", [])) if depth else None)
        _sleep(0.2)

    print("Stats:", client.get_orderbook_stats())
