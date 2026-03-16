from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd  # type: ignore

try:
    from coinbase_advanced import GRANULARITY_TO_SECONDS
except ImportError:  # pragma: no cover - package import path
    from scalping_5min_momentum.coinbase_advanced import GRANULARITY_TO_SECONDS

TIMEFRAME_MODE_STRICT = "strict_1m_on_5m"
TIMEFRAME_MODE_FIVE_ONLY = "5m_only"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BreakoutConfig:
    timeframe_mode: str = TIMEFRAME_MODE_STRICT
    signal_granularity: str = "ONE_MINUTE"
    context_granularity: str = "FIVE_MINUTE"
    reward_risk_ratio: float = 1.5
    atr_period: int = 14
    volume_window: int = 20
    min_box_atr_ratio: float = 0.8
    min_volume_ratio: float = 1.0
    risk_fraction: float = 0.01
    leverage: float = 2.0
    min_position_notional: float = 10.0
    max_position_notional: float = 500.0
    kalman_process_variance: float = 0.0005
    kalman_measurement_variance: float = 0.01


@dataclass
class BoxState:
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    high: Optional[float] = None
    low: Optional[float] = None
    range_size: Optional[float] = None
    atr: Optional[float] = None


@dataclass
class PositionState:
    is_open: bool = False
    direction: Optional[str] = None
    entry_price: float = 0.0
    base_size: float = 0.0
    notional: float = 0.0
    margin_used: float = 0.0
    leverage: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    breakout_high: Optional[float] = None
    breakout_low: Optional[float] = None
    signal_time: Optional[str] = None
    opened_at: Optional[str] = None
    order_id: Optional[str] = None


@dataclass
class PositionSizePlan:
    quantity: float
    notional: float
    margin_used: float
    risk_budget: float
    risk_per_unit: float
    stop_price: float
    take_profit_price: float

    def to_dict(self) -> dict[str, float]:
        return {
            "quantity": round(self.quantity, 8),
            "notional": round(self.notional, 8),
            "margin_used": round(self.margin_used, 8),
            "risk_budget": round(self.risk_budget, 8),
            "risk_per_unit": round(self.risk_per_unit, 8),
            "stop_price": round(self.stop_price, 8),
            "take_profit_price": round(self.take_profit_price, 8),
        }


@dataclass
class SignalDecision:
    action: str
    reason: str
    signal_time: Optional[str]
    direction: Optional[str]
    reference_price: float
    stop_reference_price: Optional[float]
    take_profit_price: Optional[float]
    box: BoxState
    indicators: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["box"] = {
            key: (round(value, 8) if isinstance(value, float) else value)
            for key, value in payload["box"].items()
        }
        payload["indicators"] = {key: round(value, 8) for key, value in self.indicators.items()}
        return payload


@dataclass
class ExitDecision:
    action: str
    reason: str
    exit_price: float

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "reason": self.reason,
            "exit_price": round(self.exit_price, 8),
        }


def normalize_config(config: BreakoutConfig) -> BreakoutConfig:
    if config.timeframe_mode == TIMEFRAME_MODE_FIVE_ONLY:
        return BreakoutConfig(
            timeframe_mode=config.timeframe_mode,
            signal_granularity=config.context_granularity,
            context_granularity=config.context_granularity,
            reward_risk_ratio=config.reward_risk_ratio,
            atr_period=config.atr_period,
            volume_window=config.volume_window,
            min_box_atr_ratio=config.min_box_atr_ratio,
            min_volume_ratio=config.min_volume_ratio,
            risk_fraction=config.risk_fraction,
            leverage=config.leverage,
            min_position_notional=config.min_position_notional,
            max_position_notional=config.max_position_notional,
            kalman_process_variance=config.kalman_process_variance,
            kalman_measurement_variance=config.kalman_measurement_variance,
        )
    return config


def required_history_bars(config: BreakoutConfig) -> int:
    normalized = normalize_config(config)
    signal_seconds = GRANULARITY_TO_SECONDS[normalized.signal_granularity]
    context_seconds = GRANULARITY_TO_SECONDS[normalized.context_granularity]
    context_multiple = max(context_seconds // signal_seconds, 1)
    warmup = max(normalized.atr_period * 3, normalized.volume_window + 3, 40)
    return warmup * context_multiple + (context_multiple * 3)


def signal_frequency(config: BreakoutConfig) -> str:
    normalized = normalize_config(config)
    return _seconds_to_frequency(GRANULARITY_TO_SECONDS[normalized.signal_granularity])


def context_frequency(config: BreakoutConfig) -> str:
    normalized = normalize_config(config)
    return _seconds_to_frequency(GRANULARITY_TO_SECONDS[normalized.context_granularity])


def build_signal_frame(candles: pd.DataFrame, config: BreakoutConfig) -> pd.DataFrame:
    normalized = normalize_config(config)
    if candles.empty:
        return pd.DataFrame()

    frame = candles.copy().sort_index()
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)

    context = _build_context_frame(frame, normalized)
    signal = frame.copy()
    signal["volume_ma"] = signal["volume"].rolling(normalized.volume_window).mean()
    kalman_state = _compute_kalman_state(
        signal["close"],
        process_variance=normalized.kalman_process_variance,
        measurement_variance=normalized.kalman_measurement_variance,
    )
    signal["kalman_state"] = kalman_state
    signal["kalman_slope"] = kalman_state.diff()

    context_seconds = GRANULARITY_TO_SECONDS[normalized.context_granularity]
    signal["context_key"] = signal.index.floor(_seconds_to_frequency(context_seconds))
    joined = signal.join(
        context[
            [
                "box_high",
                "box_low",
                "box_range",
                "box_atr",
                "box_start",
                "box_end",
            ]
        ],
        on="context_key",
        how="left",
    )
    joined["long_breakout"] = joined["close"] > joined["box_high"]
    joined["short_breakout"] = joined["close"] < joined["box_low"]
    joined["volatility_ok"] = joined["box_range"] >= joined["box_atr"] * normalized.min_box_atr_ratio
    joined["volume_ok"] = joined["volume"] >= joined["volume_ma"] * normalized.min_volume_ratio
    joined["kalman_long_ok"] = (
        (joined["close"] > joined["kalman_state"]) & (joined["kalman_slope"] > 0)
    )
    joined["kalman_short_ok"] = (
        (joined["close"] < joined["kalman_state"]) & (joined["kalman_slope"] < 0)
    )
    joined["box_start"] = pd.to_datetime(joined["box_start"], utc=True, errors="coerce")
    joined["box_end"] = pd.to_datetime(joined["box_end"], utc=True, errors="coerce")
    return joined.dropna(
        subset=[
            "volume_ma",
            "kalman_state",
            "kalman_slope",
            "box_high",
            "box_low",
            "box_range",
            "box_atr",
        ]
    ).copy()


def evaluate_signal(
    candles: pd.DataFrame,
    config: BreakoutConfig,
    position: PositionState,
) -> SignalDecision:
    feature_frame = build_signal_frame(candles, config)
    if feature_frame.empty:
        raise ValueError("Not enough candle history to evaluate breakout signals")

    latest = feature_frame.iloc[-1]
    signal_time = feature_frame.index[-1].isoformat()
    box = BoxState(
        start_time=_timestamp_or_none(latest.get("box_start")),
        end_time=_timestamp_or_none(latest.get("box_end")),
        high=float(latest["box_high"]),
        low=float(latest["box_low"]),
        range_size=float(latest["box_range"]),
        atr=float(latest["box_atr"]),
    )
    indicators = _signal_indicators(latest)

    if position.is_open:
        return SignalDecision(
            action="HOLD",
            reason="Position already open; entry signals suppressed until exit",
            signal_time=signal_time,
            direction=position.direction,
            reference_price=float(latest["close"]),
            stop_reference_price=position.stop_price,
            take_profit_price=position.take_profit_price,
            box=box,
            indicators=indicators,
        )

    failed_checks = []
    if not bool(latest["volatility_ok"]):
        failed_checks.append("volatility")
    if not bool(latest["volume_ok"]):
        failed_checks.append("volume")

    if bool(latest["long_breakout"]) and bool(latest["kalman_long_ok"]) and not failed_checks:
        stop_reference = float(latest["low"])
        take_profit = build_exit_levels(
            entry_price=float(latest["close"]),
            direction="LONG",
            stop_reference_price=stop_reference,
            reward_risk_ratio=config.reward_risk_ratio,
        )
        return SignalDecision(
            action="ENTER_LONG",
            reason="Long breakout above prior box high with Kalman, volatility, and volume confirmation",
            signal_time=signal_time,
            direction="LONG",
            reference_price=float(latest["close"]),
            stop_reference_price=stop_reference,
            take_profit_price=take_profit.take_profit_price,
            box=box,
            indicators=indicators,
        )

    if bool(latest["short_breakout"]) and bool(latest["kalman_short_ok"]) and not failed_checks:
        stop_reference = float(latest["high"])
        take_profit = build_exit_levels(
            entry_price=float(latest["close"]),
            direction="SHORT",
            stop_reference_price=stop_reference,
            reward_risk_ratio=config.reward_risk_ratio,
        )
        return SignalDecision(
            action="ENTER_SHORT",
            reason="Short breakout below prior box low with Kalman, volatility, and volume confirmation",
            signal_time=signal_time,
            direction="SHORT",
            reference_price=float(latest["close"]),
            stop_reference_price=stop_reference,
            take_profit_price=take_profit.take_profit_price,
            box=box,
            indicators=indicators,
        )

    if bool(latest["long_breakout"]) and not bool(latest["kalman_long_ok"]):
        failed_checks.append("kalman_long")
    if bool(latest["short_breakout"]) and not bool(latest["kalman_short_ok"]):
        failed_checks.append("kalman_short")
    if not bool(latest["long_breakout"]) and not bool(latest["short_breakout"]):
        failed_checks.append("breakout")

    return SignalDecision(
        action="HOLD",
        reason=f"No entry. Failed filters: {', '.join(dict.fromkeys(failed_checks))}",
        signal_time=signal_time,
        direction=None,
        reference_price=float(latest["close"]),
        stop_reference_price=None,
        take_profit_price=None,
        box=box,
        indicators=indicators,
    )


def evaluate_live_exit(
    position: PositionState,
    *,
    best_bid: float,
    best_ask: float,
) -> ExitDecision:
    if not position.is_open or position.direction is None:
        return ExitDecision(action="HOLD", reason="No open position", exit_price=0.0)
    if position.stop_price is None or position.take_profit_price is None:
        return ExitDecision(action="HOLD", reason="Position missing exit levels", exit_price=0.0)

    if position.direction == "LONG":
        if best_bid <= position.stop_price:
            return ExitDecision(
                action="EXIT",
                reason="Stop-loss hit",
                exit_price=best_bid,
            )
        if best_bid >= position.take_profit_price:
            return ExitDecision(
                action="EXIT",
                reason="Take-profit hit",
                exit_price=best_bid,
            )
    else:
        if best_ask >= position.stop_price:
            return ExitDecision(
                action="EXIT",
                reason="Stop-loss hit",
                exit_price=best_ask,
            )
        if best_ask <= position.take_profit_price:
            return ExitDecision(
                action="EXIT",
                reason="Take-profit hit",
                exit_price=best_ask,
            )
    return ExitDecision(action="HOLD", reason="Position still active", exit_price=0.0)


def build_exit_levels(
    *,
    entry_price: float,
    direction: str,
    stop_reference_price: float,
    reward_risk_ratio: float,
) -> PositionSizePlan:
    if direction == "LONG":
        risk_per_unit = entry_price - stop_reference_price
        if risk_per_unit <= 0:
            raise ValueError("Long stop reference must be below entry price")
        take_profit_price = entry_price + risk_per_unit * reward_risk_ratio
        stop_price = stop_reference_price
    else:
        risk_per_unit = stop_reference_price - entry_price
        if risk_per_unit <= 0:
            raise ValueError("Short stop reference must be above entry price")
        take_profit_price = entry_price - risk_per_unit * reward_risk_ratio
        stop_price = stop_reference_price
    return PositionSizePlan(
        quantity=0.0,
        notional=0.0,
        margin_used=0.0,
        risk_budget=0.0,
        risk_per_unit=risk_per_unit,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
    )


def calculate_position_size(
    *,
    equity: float,
    entry_price: float,
    stop_reference_price: float,
    direction: str,
    config: BreakoutConfig,
    fee_rate: float = 0.0,
) -> PositionSizePlan:
    if equity <= 0:
        raise ValueError("Equity must be positive")
    if config.leverage <= 0:
        raise ValueError("Leverage must be positive")

    exits = build_exit_levels(
        entry_price=entry_price,
        direction=direction,
        stop_reference_price=stop_reference_price,
        reward_risk_ratio=config.reward_risk_ratio,
    )
    risk_budget = equity * config.risk_fraction
    raw_quantity = risk_budget / exits.risk_per_unit
    raw_notional = raw_quantity * entry_price

    max_notional_for_cash = (equity * config.leverage) / (1 + config.leverage * max(fee_rate, 0.0))
    bounded_notional = min(raw_notional, config.max_position_notional, max_notional_for_cash)
    if bounded_notional < config.min_position_notional:
        raise ValueError("Calculated notional falls below the configured minimum")

    quantity = bounded_notional / entry_price
    margin_used = bounded_notional / config.leverage
    return PositionSizePlan(
        quantity=quantity,
        notional=bounded_notional,
        margin_used=margin_used,
        risk_budget=risk_budget,
        risk_per_unit=exits.risk_per_unit,
        stop_price=exits.stop_price,
        take_profit_price=exits.take_profit_price,
    )


def approximate_liquidation_price(entry_price: float, direction: str, leverage: float) -> float:
    if leverage <= 0:
        raise ValueError("Leverage must be positive")
    if direction == "LONG":
        return entry_price * max(0.0, 1 - 1 / leverage)
    return entry_price * (1 + 1 / leverage)


def mark_to_market_pnl(position: PositionState, mark_price: float) -> float:
    if not position.is_open or position.direction is None:
        return 0.0
    if position.direction == "LONG":
        return position.base_size * (mark_price - position.entry_price)
    return position.base_size * (position.entry_price - mark_price)


def _build_context_frame(candles: pd.DataFrame, config: BreakoutConfig) -> pd.DataFrame:
    signal_seconds = GRANULARITY_TO_SECONDS[config.signal_granularity]
    context_seconds = GRANULARITY_TO_SECONDS[config.context_granularity]

    if signal_seconds == context_seconds:
        context = candles.copy()
    else:
        context = candles.resample(
            _seconds_to_frequency(context_seconds),
            label="left",
            closed="left",
        ).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        context = context.dropna()

    context["context_atr"] = _compute_atr(context, config.atr_period)
    context["box_high"] = context["high"].shift(1)
    context["box_low"] = context["low"].shift(1)
    context["box_range"] = context["box_high"] - context["box_low"]
    context["box_atr"] = context["context_atr"].shift(1)
    context["box_start"] = context.index.to_series().shift(1)
    context["box_end"] = context["box_start"] + pd.to_timedelta(context_seconds, unit="s")
    return context


def _compute_kalman_state(
    close: pd.Series,
    *,
    process_variance: float,
    measurement_variance: float,
) -> pd.Series:
    if close.empty:
        return close.copy()

    estimates: list[float] = []
    estimate = float(close.iloc[0])
    error_covariance = 1.0
    for raw_value in close.astype(float):
        error_covariance += process_variance
        kalman_gain = error_covariance / (error_covariance + measurement_variance)
        estimate = estimate + kalman_gain * (float(raw_value) - estimate)
        error_covariance = (1 - kalman_gain) * error_covariance
        estimates.append(estimate)
    return pd.Series(estimates, index=close.index, dtype=float)


def _compute_atr(frame: pd.DataFrame, period: int) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def _signal_indicators(row: pd.Series) -> dict[str, float]:
    return {
        "close": float(row["close"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "volume": float(row["volume"]),
        "volume_ma": float(row["volume_ma"]),
        "kalman_state": float(row["kalman_state"]),
        "kalman_slope": float(row["kalman_slope"]),
        "box_high": float(row["box_high"]),
        "box_low": float(row["box_low"]),
        "box_range": float(row["box_range"]),
        "box_atr": float(row["box_atr"]),
    }


def _seconds_to_frequency(seconds: int) -> str:
    return f"{seconds}s"


def _timestamp_or_none(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


__all__ = [
    "SCHEMA_VERSION",
    "TIMEFRAME_MODE_FIVE_ONLY",
    "TIMEFRAME_MODE_STRICT",
    "BoxState",
    "BreakoutConfig",
    "ExitDecision",
    "PositionSizePlan",
    "PositionState",
    "SignalDecision",
    "approximate_liquidation_price",
    "build_exit_levels",
    "build_signal_frame",
    "calculate_position_size",
    "context_frequency",
    "evaluate_live_exit",
    "evaluate_signal",
    "mark_to_market_pnl",
    "normalize_config",
    "required_history_bars",
    "signal_frequency",
]
