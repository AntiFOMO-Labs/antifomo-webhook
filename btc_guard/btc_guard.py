# btc_guard.py
#
# BTCGuard v2.1 (Per-Coin)
# Двухслойная защита:
# 1) Глобальное состояние BTC: IDLE / WARNING / ALERT (по его импульсу)
# 2) Локальное решение по монете: по btc_corr решаем, блокировать ли шорт
#
# Интерфейс:
#   guard = BTCGuard()
#   guard.on_btc_tick(now_ts=time.time(), close_price=price_btc)
#   decision = guard.handle_signal(symbol, side, btc_corr, now_ts, ts_str, signal_name)
#
#   if decision == BTCDecision.DEFER:  -> сигнал откладываем (BTC тащит и монета сильно коррелирует)
#   if decision == BTCDecision.ALLOW:  -> пропускаем сигнал в AntiFOMO

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from datetime import timedelta


class BTCGuardState(Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    ALERT = "ALERT"


class BTCDecision(Enum):
    ALLOW = "ALLOW"   # пропускаем сигнал дальше
    DEFER = "DEFER"   # откладываем сигнал (BTC импульс + сильная корреляция по монете)
    BLOCK = "BLOCK"   # жёстко игнорируем (пока не используем)


@dataclass
class BTCGuardConfig:
    # Пороговые значения для импульса BTC
    warning_threshold_5m: float = 0.7    # % за 5 минут
    warning_threshold_15m: float = 1.5   # % за 15 минут
    warning_threshold_30m: float = 2.0   # % за 30 минут

    alert_threshold_5_15: float = 1.8    # % за 5–15 минут для ALERT

    # Порог по корреляции для блокировки шорта
    corr_block_threshold: float = 0.7    # выше этого — монета ходит за BTC, шорт опасен

    # TTL для отложенных сигналов
    deferred_ttl_min: int = 45

    # окна для расчёта % изменения BTC
    window_5m: int = 5
    window_15m: int = 15
    window_30m: int = 30


@dataclass
class BTCPricePoint:
    ts: float      # timestamp в секундах (time.time())
    price: float


@dataclass
class DeferredItem:
    symbol: str
    side: str
    signal: str
    created_at: float
    raw_ts_str: str


class BTCGuard:
    """
    BTCGuard: умная защита AntiFOMO от входов в шорт против сильного импульса BTC.

    - Следит за движением BTC во времени (по тикерам BTCUSDT с TradingView).
    - Переключает состояния: IDLE / WARNING / ALERT.
    - В ALERT:
        - если btc_corr монеты >= corr_block_threshold и side == SHORT → DEFER
        - если btc_corr ниже порога → ALLOW (монета уже отвалилась от BTC).
    """

    def __init__(self, config: Optional[BTCGuardConfig] = None) -> None:
        self.config = config or BTCGuardConfig()
        self.state: BTCGuardState = BTCGuardState.IDLE

        # История BTC (последние ~60 минут)
        self._btc_history: List[BTCPricePoint] = []

        # Отложенные сигналы (на будущее, пока только считаем)
        self._deferred: List[DeferredItem] = []

    # ─────────────────────────────────────────
    #           ПУБЛИЧНОЕ API
    # ─────────────────────────────────────────

    def on_btc_tick(self, now_ts: float, close_price: float) -> None:
        """
        Обновление цены BTC.
        Вызывается при каждом алерте по BTC (1m, 5m – как настроишь).
        now_ts — time.time() сервера.
        """
        self._add_btc_point(now_ts, close_price)
        self._update_state(now_ts)

    def handle_signal(
        self,
        symbol: str,
        side: str,
        btc_corr: float,
        now_ts: float,
        ts_str: str = "",
        signal_name: str = "",
    ) -> BTCDecision:
        """
        Решение по конкретному сигналу по монете.

        - symbol      — пара (BINANCE:BOMEUSDT.P и т.п.)
        - side        — "SHORT" / "LONG" / "?"
        - btc_corr    — корреляция монеты с BTC (0..1 из JSON индикатора)
        - now_ts      — time.time() сервера
        - ts_str      — строковое время из TradingView (для логов)
        - signal_name — индикатор (A1, A12, A9L и т.д.)
        """
        self._cleanup_deferred(now_ts)

        # Нас интересуют только SHORT-сигналы
        if side != "SHORT":
            return BTCDecision.ALLOW

        # Если BTC не в ALERT — ничего не блокируем
        if self.state != BTCGuardState.ALERT:
            return BTCDecision.ALLOW

        # BTC в ALERT → смотрим, насколько монета завязана на него
        try:
            corr = float(btc_corr)
        except Exception:
            corr = 0.0

        if corr >= self.config.corr_block_threshold:
            # Монета сильно коррелирует с BTC → шорт опасен, откладываем
            self._defer_signal(symbol, side, signal_name, now_ts, ts_str)
            print(
                f"[BTCGuard] DEFER symbol={symbol}, side={side}, "
                f"signal={signal_name}, corr={corr:.2f}, state={self.state.value}"
            )
            return BTCDecision.DEFER
        else:
            # Монета отстаёт от BTC → даём зелёный свет для шорта
            print(
                f"[BTCGuard] ALLOW (weak corr) symbol={symbol}, side={side}, "
                f"signal={signal_name}, corr={corr:.2f}, state={self.state.value}"
            )
            return BTCDecision.ALLOW

    def get_state(self) -> BTCGuardState:
        return self.state

    def get_deferred_summary(self) -> int:
        """Сколько сигналов сейчас лежит в отложенных (для телеметрии/логов)."""
        return len(self._deferred)

    # ─────────────────────────────────────────
    #         ВНУТРЕННЯЯ ЛОГИКА BTC
    # ─────────────────────────────────────────

    def _add_btc_point(self, now_ts: float, price: float) -> None:
        self._btc_history.append(BTCPricePoint(ts=now_ts, price=price))
        cutoff = now_ts - 60 * 60  # 60 минут
        self._btc_history = [p for p in self._btc_history if p.ts >= cutoff]

    def _update_state(self, now_ts: float) -> None:
        c = self.config
        r5 = self._get_change_percent(c.window_5m, now_ts)
        r15 = self._get_change_percent(c.window_15m, now_ts)
        r30 = self._get_change_percent(c.window_30m, now_ts)

        prev_state = self.state

        # 1) Проверка на вход в ALERT
        if self._should_enter_alert(r5, r15):
            self.state = BTCGuardState.ALERT
        else:
            # 2) Проверка выхода из ALERT
            if self.state == BTCGuardState.ALERT and self._can_exit_alert(now_ts):
                self.state = BTCGuardState.IDLE

            # 3) WARNING (если не ALERT)
            if self.state != BTCGuardState.ALERT:
                if self._should_enter_warning(r5, r15, r30):
                    self.state = BTCGuardState.WARNING
                else:
                    self.state = BTCGuardState.IDLE

        if prev_state != self.state:
            print(
                f"[BTCGuard] STATE CHANGE {prev_state.value} → {self.state.value}; "
                f"r5={r5}, r15={r15}, r30={r30}"
            )

    def _get_change_percent(self, minutes: int, now_ts: float) -> Optional[float]:
        """% изменения цены BTC за N минут."""
        if not self._btc_history:
            return None

        cutoff = now_ts - minutes * 60
        past_candidates = [p for p in self._btc_history if p.ts <= cutoff]
        if not past_candidates:
            return None

        past = past_candidates[0]
        current = self._btc_history[-1]
        if past.price == 0:
            return None

        return (current.price - past.price) / past.price * 100.0

    def _should_enter_warning(
        self,
        r5: Optional[float],
        r15: Optional[float],
        r30: Optional[float],
    ) -> bool:
        c = self.config
        return (
            (r5 is not None and r5 >= c.warning_threshold_5m)
            or (r15 is not None and r15 >= c.warning_threshold_15m)
            or (r30 is not None and r30 >= c.warning_threshold_30m)
        )

    def _should_enter_alert(
        self,
        r5: Optional[float],
        r15: Optional[float],
    ) -> bool:
        c = self.config
        return (
            (r5 is not None and r5 >= c.alert_threshold_5_15)
            or (r15 is not None and r15 >= c.alert_threshold_5_15)
        )

    def _can_exit_alert(self, now_ts: float) -> bool:
        """
        Выход из ALERT — когда BTC успокоился:
        % изменений за 5 и 15 минут стали маленькими.
        """
        c = self.config
        r5 = self._get_change_percent(c.window_5m, now_ts)
        r15 = self._get_change_percent(c.window_15m, now_ts)

        if r5 is None or r15 is None:
            return False

        calm_threshold = 0.5  # % — можно вынести в конфиг
        return abs(r5) < calm_threshold and abs(r15) < calm_threshold

    # ─────────────────────────────────────────
    #         ОТЛОЖЕННЫЕ СИГНАЛЫ
    # ─────────────────────────────────────────

    def _defer_signal(
        self,
        symbol: str,
        side: str,
        signal_name: str,
        now_ts: float,
        ts_str: str,
    ) -> None:
        self._deferred.append(
            DeferredItem(
                symbol=symbol,
                side=side,
                signal=signal_name,
                created_at=now_ts,
                raw_ts_str=ts_str,
            )
        )

    def _cleanup_deferred(self, now_ts: float) -> None:
        ttl_sec = self.config.deferred_ttl_min * 60
        self._deferred = [
            d for d in self._deferred if now_ts - d.created_at <= ttl_sec
        ]
