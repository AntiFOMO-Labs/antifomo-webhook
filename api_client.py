#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

BINANCE_FAPI_URL = "https://fapi.binance.com"
BINANCE_FAPI_TESTNET_URL = "https://testnet.binancefuture.com"


class BinanceFuturesClient:
    """Обёртка над python-binance для AntiFOMO."""

    def __init__(self, testnet: bool = False, recv_window: int = 5000):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET are not set in .env")

        self.testnet = testnet
        self.recv_window = recv_window

        self.client = Client(api_key, api_secret, testnet=testnet)

        if testnet:
            self.client.FUTURES_URL = BINANCE_FAPI_TESTNET_URL
        else:
            self.client.FUTURES_URL = BINANCE_FAPI_URL

        self.sync_timestamp(initial=True)

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
        try:
            data = self.client.futures_symbol_ticker(symbol=symbol)
            return float(data["price"])
        except Exception as e:
            self._handle_exception(e, f"get_price({symbol})")
            return None

    def get_order_book(self, symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
        try:
            limit = max(5, min(limit, 500))
            depth = self.client.futures_order_book(symbol=symbol, limit=limit)
            return depth
        except Exception as e:
            self._handle_exception(e, f"get_order_book({symbol})")
            return None

    def get_liquidations(
        self,
        symbol: Optional[str] = None,
        limit: int = 50,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> Optional[list]:
        params: Dict[str, Any] = {"limit": max(1, min(limit, 1000))}
        if symbol:
            params["symbol"] = symbol
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
