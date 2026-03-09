from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd  # type: ignore


@dataclass
class ScalpingConfig:
    fast_ema: int = 8
    slow_ema: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    volume_window: int = 20
    min_rsi_entry: float = 55.0
    max_rsi_entry: float = 78.0
    exit_rsi: float = 48.0
    min_volume_ratio: float = 0.8
    stop_atr_multiple: float = 1.2
    take_profit_atr_multiple: float = 1.8
    trailing_atr_multiple: float = 1.0
    risk_fraction: float = 0.02
    min_quote_notional: float = 10.0
    max_quote_notional: float = 100.0


@dataclass
class PositionState:
    is_open: bool = False
    entry_price: float = 0.0
    base_size: float = 0.0
    quote_size: float = 0.0
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    opened_at: Optional[str] = None
    order_id: Optional[str] = None


@dataclass
class StrategyDecision:
    action: str
    reason: str
    latest_price: float
    stop_price: Optional[float]
    take_profit_price: Optional[float]
    indicators: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["indicators"] = {key: round(value, 8) for key, value in self.indicators.items()}
        return payload


def required_history_bars(config: ScalpingConfig) -> int:
    return max(
        config.slow_ema + 2,
        config.rsi_period + 2,
        config.atr_period + 2,
        config.volume_window + 2,
    )


def add_indicators(candles: pd.DataFrame, config: ScalpingConfig) -> pd.DataFrame:
    frame = candles.copy()
    frame["ema_fast"] = frame["close"].ewm(span=config.fast_ema, adjust=False).mean()
    frame["ema_slow"] = frame["close"].ewm(span=config.slow_ema, adjust=False).mean()
    frame["rsi"] = _compute_rsi(frame["close"], config.rsi_period)
    frame["atr"] = _compute_atr(frame, config.atr_period)
    frame["volume_ma"] = frame["volume"].rolling(config.volume_window).mean()
    frame["vwap"] = (frame["close"] * frame["volume"]).cumsum() / frame["volume"].cumsum().clip(lower=1e-12)
    return frame.dropna().copy()


def suggest_quote_order_size(
    available_quote_balance: float,
    config: ScalpingConfig,
    quote_min_size: float,
) -> float:
    if available_quote_balance <= 0:
        return 0.0
    proposed = min(available_quote_balance * config.risk_fraction, config.max_quote_notional)
    proposed = max(proposed, config.min_quote_notional)
    proposed = min(proposed, available_quote_balance)
    if proposed < quote_min_size:
        return 0.0
    return proposed


def evaluate_scalping_decision(
    candles: pd.DataFrame,
    config: ScalpingConfig,
    position: PositionState,
    *,
    best_bid: Optional[float] = None,
    best_ask: Optional[float] = None,
) -> StrategyDecision:
    frame = add_indicators(candles, config)
    if len(frame) < 2:
        raise ValueError("Not enough candle history to evaluate the strategy")

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    last_close = float(latest["close"])
    execution_bid = best_bid or last_close
    execution_ask = best_ask or last_close
    atr = float(latest["atr"])

    indicators = {
        "close": last_close,
        "ema_fast": float(latest["ema_fast"]),
        "ema_slow": float(latest["ema_slow"]),
        "rsi": float(latest["rsi"]),
        "atr": atr,
        "vwap": float(latest["vwap"]),
        "volume": float(latest["volume"]),
        "volume_ma": float(latest["volume_ma"]),
        "best_bid": float(execution_bid),
        "best_ask": float(execution_ask),
    }

    if not position.is_open:
        checks = {
            "trend": last_close > float(latest["ema_fast"]) > float(latest["ema_slow"]),
            "momentum": config.min_rsi_entry <= float(latest["rsi"]) <= config.max_rsi_entry,
            "vwap": last_close >= float(latest["vwap"]),
            "volume": float(latest["volume"]) >= float(latest["volume_ma"]) * config.min_volume_ratio,
            "ema_slope": float(latest["ema_fast"]) > float(previous["ema_fast"]),
        }
        if all(checks.values()):
            stop_price = execution_ask - atr * config.stop_atr_multiple
            take_profit_price = execution_ask + atr * config.take_profit_atr_multiple
            return StrategyDecision(
                action="BUY",
                reason="Long scalp entry: EMA trend, RSI, VWAP, and volume filters aligned",
                latest_price=execution_ask,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                indicators=indicators,
            )

        failed_checks = ", ".join(name for name, passed in checks.items() if not passed)
        return StrategyDecision(
            action="HOLD",
            reason=f"No entry. Failed filters: {failed_checks}",
            latest_price=execution_ask,
            stop_price=None,
            take_profit_price=None,
            indicators=indicators,
        )

    trailing_stop = max(
        position.stop_price or float("-inf"),
        execution_bid - atr * config.trailing_atr_multiple,
    )
    take_profit = position.take_profit_price or (
        position.entry_price + atr * config.take_profit_atr_multiple
    )

    if execution_bid <= trailing_stop:
        return StrategyDecision(
            action="SELL",
            reason="Exit on ATR trailing stop",
            latest_price=execution_bid,
            stop_price=trailing_stop,
            take_profit_price=take_profit,
            indicators=indicators,
        )
    if execution_bid >= take_profit:
        return StrategyDecision(
            action="SELL",
            reason="Exit on ATR take-profit",
            latest_price=execution_bid,
            stop_price=trailing_stop,
            take_profit_price=take_profit,
            indicators=indicators,
        )
    if float(latest["rsi"]) <= config.exit_rsi:
        return StrategyDecision(
            action="SELL",
            reason="Exit on RSI momentum fade",
            latest_price=execution_bid,
            stop_price=trailing_stop,
            take_profit_price=take_profit,
            indicators=indicators,
        )
    if last_close < float(latest["ema_fast"]) or last_close < float(latest["vwap"]):
        return StrategyDecision(
            action="SELL",
            reason="Exit on intraday trend breakdown",
            latest_price=execution_bid,
            stop_price=trailing_stop,
            take_profit_price=take_profit,
            indicators=indicators,
        )

    return StrategyDecision(
        action="HOLD",
        reason="Position still valid; keep trailing stop updated",
        latest_price=execution_bid,
        stop_price=trailing_stop,
        take_profit_price=take_profit,
        indicators=indicators,
    )


def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    relative_strength = average_gain / average_loss.where(average_loss != 0)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(~((average_gain == 0) & (average_loss == 0)), 50.0)
    rsi = rsi.where(~((average_gain > 0) & (average_loss == 0)), 100.0)
    rsi = rsi.where(~((average_gain == 0) & (average_loss > 0)), 0.0)
    return rsi


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


__all__ = [
    "PositionState",
    "ScalpingConfig",
    "StrategyDecision",
    "add_indicators",
    "evaluate_scalping_decision",
    "required_history_bars",
    "suggest_quote_order_size",
]
