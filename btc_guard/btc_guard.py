# btc_guard/btc_guard.py
#
# BTCGuard v2.2-P(LQ)
#
# Двухслойная защита:
# 1) Глобальное состояние BTC: IDLE / WARNING / ALERT (по импульсу BTC)
# 2) Локальное решение по монете: по btc_corr и liq_level решаем, блокировать ли SHORT
#
# Интерфейс:
#   guard = BTCGuard()
#
#   # тик по BTCUSDT (1m / 5m)
#   guard.on_btc_tick(now_ts=time.time(), close_price=btc_price)
#
#   # сигнал по монете
#   decision = guard.handle_signal(
#       symbol="BINANCE:BOMEUSDT.P",
#       side="SHORT",
#       btc_corr=0.23,
#       liq_level="HIGH",   # "HIGH" / "MEDIUM" / "LOW" / "ULTRA_LOW" / None
#       now_ts=time.time(),
#       ts_str="2025-12-04 12:34:56",
#       signal_name="A12",
#   )
#
#   if decision == BTCDecision.DEFER:
#       # глушим сигнал (не запускаем AntiFOMO по этой монете)
#   if decision == BTCDecision.ALLOW:
#       # пускаем сигнал в обычную логику AntiFOMO


from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple


class BTCGuardState(Enum):
    IDLE = "IDLE"
    WARNING = "WARNING"
    ALERT = "ALERT"


class BTCDecision(Enum):
    ALLOW = "ALLOW"   # пропускаем сигнал дальше
    DEFER = "DEFER"   # откладываем сигнал (BTC импульс + риск по монете)
    BLOCK = "BLOCK"   # жёстко игнорируем (пока не используем отдельно)


@dataclass
class BTCGuardConfig:
    # Пороговые значения для импульса BTC (на глобальном уровне)
    warning_threshold_5m: float = 0.7    # % за 5 минут
    warning_threshold_15m: float = 1.5   # % за 15 минут
    warning_threshold_30m: float = 2.0   # % за 30 минут

    alert_threshold_5_15: float = 1.8    # % за 5–15 минут для ALERT

    # Порог по корреляции для жёсткой блокировки
    corr_block_threshold: float = 0.70   # corr_now >= 0.70 → блок

    # "Слабая" корреляция (используется для логики "была слабой → стала средней")
    weak_corr_max: float = 0.40         # ниже этого считаем монету слабой

    # Порог "перепривязки": если была слабой, а стала >= этого → блок
    rebind_corr_threshold: float = 0.55

    # Порог для скачка корреляции
    delta_corr_block_threshold: float = 0.25  # рост corr_now - corr_prev >= 0.25 → блок

    # TTL для отложенных сигналов (секунды)
    deferred_ttl_sec: int = 45 * 60

    # Окна для расчёта % изменения BTC
    window_5m: int = 5
    window_15m: int = 15
    window_30m: int = 30


@dataclass
class BTCPricePoint:
    ts: float      # timestamp (time.time())
    price: float


@dataclass
class DeferredItem:
    symbol: str
    side: str
    signal: str
    created_at: float
    raw_ts_str: str
    btc_corr: float
    liq_level: str
    btc_state: str


@dataclass
class SymbolCorrState:
    last_corr: Optional[float] = None
    prev_corr: Optional[float] = None
    last_update_ts: Optional[float] = None


class BTCGuard:
    """
    BTCGuard v2.2-P(LQ):

    - Следит за движением BTC (по тикерам BTCUSDT).
    - Переключает состояния: IDLE / WARNING / ALERT.
    - В ALERT принимает решение по КАЖДОЙ монете:
        - по btc_corr (текущая и предыдущая),
        - по delta_corr (насколько быстро выросла связь),
        - по liq_level (LOW / ULTRA_LOW → блок),
      и решает:
        - DEFER (глушим сигнал, не даём шорт в AntiFOMO),
        - ALLOW (пускаем дальше, даже в ALERT).
    """

    def __init__(self, config: Optional[BTCGuardConfig] = None) -> None:
        self.config = config or BTCGuardConfig()
        self.state: BTCGuardState = BTCGuardState.IDLE

        # История BTC за ~60 минут
        self._btc_history: List[BTCPricePoint] = []

        # Отложенные сигналы (на будущее, пока используется только для телеметрии)
        self._deferred: List[DeferredItem] = []

        # Состояние корреляции по символам
        self._symbol_corr: Dict[str, SymbolCorrState] = {}

    # ─────────────────────────────────────────
    #           ПУБЛИЧНОЕ API
    # ─────────────────────────────────────────

    def on_btc_tick(self, now_ts: float, close_price: float) -> None:
        """
        Обновление цены BTC.
        Вызывается при каждом алерте по BTC (1m, 5m — как настроено).
        """
        self._add_btc_point(now_ts, close_price)
        self._update_state(now_ts)

    def handle_signal(
        self,
        symbol: str,
        side: str,
        btc_corr: float,
        liq_level: Optional[str],
        now_ts: float,
        ts_str: str = "",
        signal_name: str = "",
    ) -> BTCDecision:
        """
        Решение по конкретному сигналу по монете.

        :param symbol:     пара, например "BINANCE:BOMEUSDT.P"
        :param side:       "SHORT" / "LONG" / "?"
        :param btc_corr:   текущая корреляция монеты с BTC (0..1)
        :param liq_level:  "HIGH" / "MEDIUM" / "LOW" / "ULTRA_LOW" / None
        :param now_ts:     time.time() сервера
        :param ts_str:     строковое время (для логов)
        :param signal_name: "A1"/"A12"/"A9L" и т.п.
        """
        self._cleanup_deferred(now_ts)

        # Нас интересуют только шортовые сигналы
        if side.upper() != "SHORT":
            return BTCDecision.ALLOW

        # Если BTC не в ALERT — не вмешиваемся, пропускаем
        if self.state != BTCGuardState.ALERT:
            return BTCDecision.ALLOW

        # Нормализуем значения
        try:
            corr_now = float(btc_corr)
        except Exception:
            corr_now = 0.0

        liq = (liq_level or "UNKNOWN").upper().strip()

        # Обновляем историю корреляций по символу
        prev_corr, delta_corr = self._update_symbol_corr(symbol, corr_now, now_ts)

        # Правила блокировки
        blocked, reason = self._should_block_symbol(
            symbol=symbol,
            corr_now=corr_now,
            prev_corr=prev_corr,
            delta_corr=delta_corr,
            liq_level=liq,
        )

        if blocked:
            # Откладываем сигнал (DEFER)
            self._defer_signal(
                symbol=symbol,
                side=side,
                signal_name=signal_name,
                now_ts=now_ts,
                ts_str=ts_str,
                btc_corr=corr_now,
                liq_level=liq,
            )
            print(
                f"[BTCGuard][DEFER] symbol={symbol}, side={side}, "
                f"signal={signal_name}, corr_now={corr_now:.2f}, "
                f"prev_corr={prev_corr if prev_corr is not None else 'None'}, "
                f"delta_corr={delta_corr if delta_corr is not None else 'None'}, "
                f"liq={liq}, state={self.state.value}, reason={reason}"
            )
            return BTCDecision.DEFER

        # Если не заблокировали — разрешаем
        print(
            f"[BTCGuard][ALLOW] symbol={symbol}, side={side}, "
            f"signal={signal_name}, corr_now={corr_now:.2f}, "
            f"prev_corr={prev_corr if prev_corr is not None else 'None'}, "
            f"delta_corr={delta_corr if delta_corr is not None else 'None'}, "
            f"liq={liq}, state={self.state.value}, reason='OK'"
        )
        return BTCDecision.ALLOW

    def get_state(self) -> BTCGuardState:
        return self.state

    def get_deferred_summary(self) -> int:
        """Сколько сигналов сейчас лежит в отложенных (для телеметрии)."""
        return len(self._deferred)

    def get_symbol_corr_state(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Возвращает (last_corr, prev_corr) для символа.
        Можно использовать для диагностики / логов.
        """
        s = self._symbol_corr.get(symbol)
        if not s:
            return None, None
        return s.last_corr, s.prev_corr

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
                f"[BTCGuard][STATE] {prev_state.value} -> {self.state.value} | "
                f"r5={r5:.2f if r5 is not None else 'None'}, "
                f"r15={r15:.2f if r15 is not None else 'None'}, "
                f"r30={r30:.2f if r30 is not None else 'None'}"
            )

    def _get_change_percent(self, minutes: int, now_ts: float) -> Optional[float]:
        """
        % изменения цены BTC за N минут.
        """
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

        calm_threshold = 0.5  # % — порог "спокойствия"
        return abs(r5) < calm_threshold and abs(r15) < calm_threshold

    # ─────────────────────────────────────────
    #       ОТСЛЕЖИВАНИЕ КОРРЕЛЯЦИИ ПО МОНЕТАМ
    # ─────────────────────────────────────────

    def _update_symbol_corr(
        self,
        symbol: str,
        corr_now: float,
        now_ts: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Обновляем историю корреляции по символу.
        Возвращаем (prev_corr, delta_corr).
        """
        state = self._symbol_corr.get(symbol)
        if state is None:
            state = SymbolCorrState()
            self._symbol_corr[symbol] = state

        prev_corr = state.last_corr
        state.prev_corr = prev_corr
        state.last_corr = corr_now
        state.last_update_ts = now_ts

        delta_corr: Optional[float] = None
        if prev_corr is not None:
            delta_corr = corr_now - prev_corr

        return prev_corr, delta_corr

    # ─────────────────────────────────────────
    #        ЛОГИКА БЛОКИРОВКИ ПО МОНЕТЕ
    # ─────────────────────────────────────────

    def _should_block_symbol(
        self,
        symbol: str,
        corr_now: float,
        prev_corr: Optional[float],
        delta_corr: Optional[float],
        liq_level: str,
    ) -> Tuple[bool, str]:
        """
        Основная логика блокировки/разрешения по монете в ALERT.
        Версия B + Liquidity Layer.
        """
        c = self.config
        liq = liq_level.upper()

        # 1) Низкая ликвидность → блок в ALERT
        if liq in ("LOW", "ULTRA_LOW"):
            return True, f"Low liquidity ({liq}) in ALERT"

        # 2) Жёсткий порог по текущей корреляции
        if corr_now >= c.corr_block_threshold:
            return True, f"corr_now >= {c.corr_block_threshold:.2f}"

        # 3) Монета была слабой, но поднялась до rebind_corr_threshold
        if prev_corr is not None:
            if prev_corr < c.weak_corr_max and corr_now >= c.rebind_corr_threshold:
                return True, (
                    f"prev_corr={prev_corr:.2f} < {c.weak_corr_max:.2f} "
                    f"and corr_now={corr_now:.2f} >= {c.rebind_corr_threshold:.2f}"
                )

        # 4) Быстрый скачок корреляции
        if delta_corr is not None and delta_corr >= c.delta_corr_block_threshold:
            return True, f"delta_corr={delta_corr:.2f} >= {c.delta_corr_block_threshold:.2f}"

        # Иначе — шорт по монете в ALERT разрешаем
        return False, "allowed in ALERT (weak corr + sufficient liquidity)"

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
        btc_corr: float,
        liq_level: str,
    ) -> None:
        self._deferred.append(
            DeferredItem(
                symbol=symbol,
                side=side,
                signal=signal_name,
                created_at=now_ts,
                raw_ts_str=ts_str,
                btc_corr=btc_corr,
                liq_level=liq_level,
                btc_state=self.state.value,
            )
        )

    def _cleanup_deferred(self, now_ts: float) -> None:
        ttl = float(self.config.deferred_ttl_sec)
        self._deferred = [
            d for d in self._deferred if now_ts - d.created_at <= ttl
        ]
