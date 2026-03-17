from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional

import pandas as pd  # type: ignore

from scalping_5min_momentum.scalping_strategy import (
    BreakoutConfig,
    approximate_liquidation_price,
    build_signal_frame,
    calculate_position_size,
)

EntryGate = Callable[[pd.Series, pd.Timestamp], bool]

SIDE_MODE_BOTH = "both"
SIDE_MODE_LONG_ONLY = "long_only"
SIDE_MODE_SHORT_ONLY = "short_only"
STOP_FAMILY_BREAKOUT_CANDLE = "breakout_candle"
STOP_FAMILY_BOX_EDGE = "box_edge"
STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX = "worse_of_candle_and_box"
STOP_FAMILY_STRUCTURAL_ATR_BUFFER = "structural_atr_buffer"


@dataclass
class BacktestConfig:
    starting_cash: float = 10000.0
    leverage: float = 2.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0003
    entry_liquidity: str = "taker"
    exit_liquidity: str = "taker"
    slippage_bps: float = 2.0


@dataclass(frozen=True)
class ResearchVariant:
    name: str = "baseline"
    side_mode: str = SIDE_MODE_BOTH
    blocked_utc_hours: tuple[int, ...] = ()
    cooldown_minutes: int = 0
    one_trade_per_box: bool = False
    min_breakout_distance_box_ratio: float = 0.0
    kalman_slope_threshold: float = 0.0
    ml_gate_enabled: bool = False
    ml_gate_threshold: float = 0.0
    long_reward_risk_ratio: Optional[float] = None
    short_reward_risk_ratio: Optional[float] = None
    long_min_box_atr_ratio: Optional[float] = None
    short_min_box_atr_ratio: Optional[float] = None
    long_min_volume_ratio: Optional[float] = None
    short_min_volume_ratio: Optional[float] = None
    long_min_breakout_distance_box_ratio: Optional[float] = None
    short_min_breakout_distance_box_ratio: Optional[float] = None
    long_kalman_slope_threshold: Optional[float] = None
    short_kalman_slope_threshold: Optional[float] = None
    stop_family: str = STOP_FAMILY_BREAKOUT_CANDLE
    stop_buffer_atr: float = 0.0
    time_stop_bars: Optional[int] = None
    breakeven_trigger_r: Optional[float] = None


@dataclass
class AssetBacktestResult:
    product_id: str
    granularity: str
    starting_cash: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades_count: int
    win_rate_pct: float
    profit_factor: float
    exposure_pct: float
    candles_count: int
    fees_paid: float
    leverage: float
    trades_frame: pd.DataFrame
    equity_curve: pd.DataFrame

    def summary(self) -> dict[str, float | int | str]:
        return {
            "product_id": self.product_id,
            "granularity": self.granularity,
            "starting_cash": round(self.starting_cash, 2),
            "final_equity": round(self.final_equity, 2),
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "trades_count": self.trades_count,
            "win_rate_pct": round(self.win_rate_pct, 2),
            "profit_factor": round(self.profit_factor, 4),
            "exposure_pct": round(self.exposure_pct, 2),
            "candles_count": self.candles_count,
            "fees_paid": round(self.fees_paid, 4),
            "leverage": round(self.leverage, 2),
        }


def run_backtest_for_asset(
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    backtest_config: BacktestConfig,
    granularity: str,
    *,
    feature_frame: Optional[pd.DataFrame] = None,
    entry_gate: Optional[EntryGate] = None,
    entry_start_time: Optional[pd.Timestamp] = None,
    research_variant: Optional[ResearchVariant] = None,
    signal_probabilities: Optional[dict[str, float]] = None,
) -> AssetBacktestResult:
    strategy_config = _strategy_with_backtest_leverage(strategy_config, backtest_config.leverage)
    research_variant = research_variant or ResearchVariant()
    if feature_frame is None:
        feature_frame = build_signal_frame(candles, strategy_config)
    else:
        feature_frame = feature_frame.copy()
    if len(feature_frame) < 3:
        raise ValueError(f"Not enough candles to backtest {product_id}")
    if entry_start_time is not None:
        entry_start_time = _coerce_utc_timestamp(entry_start_time)

    slippage_rate = backtest_config.slippage_bps / 10000.0
    entry_fee_rate = _fee_rate_for_liquidity(
        maker_rate=backtest_config.maker_fee_rate,
        taker_rate=backtest_config.taker_fee_rate,
        liquidity=backtest_config.entry_liquidity,
    )
    exit_fee_rate = _fee_rate_for_liquidity(
        maker_rate=backtest_config.maker_fee_rate,
        taker_rate=backtest_config.taker_fee_rate,
        liquidity=backtest_config.exit_liquidity,
    )
    cash = backtest_config.starting_cash
    pending_entry: Optional[dict[str, float | str]] = None
    position: Optional[dict[str, float | str]] = None
    trades: list[dict[str, float | str]] = []
    equity_rows: list[dict[str, float | str]] = []
    exposure_bars = 0
    total_fees_paid = 0.0
    last_stop_exit_at: dict[str, pd.Timestamp] = {}
    used_box_ids: set[str] = set()

    for index in range(1, len(feature_frame)):
        current = feature_frame.iloc[index]
        timestamp = feature_frame.index[index]

        if pending_entry is not None and position is None:
            direction = str(pending_entry["direction"])
            open_price = float(current["open"])
            fill_price = _apply_entry_slippage(open_price, direction, slippage_rate)
            stop_reference = float(pending_entry["stop_reference_price"])
            try:
                size_plan = calculate_position_size(
                    equity=cash,
                    entry_price=fill_price,
                    stop_reference_price=stop_reference,
                    direction=direction,
                    config=_config_with_reward_risk(
                        strategy_config,
                        float(pending_entry["reward_risk_ratio"]),
                    ),
                    fee_rate=entry_fee_rate,
                )
            except ValueError:
                pending_entry = None
                equity_rows.append({"timestamp": timestamp.isoformat(), "equity": cash})
                continue

            quantity = size_plan.quantity
            notional = size_plan.notional
            entry_fee = notional * entry_fee_rate
            cash -= entry_fee
            total_fees_paid += entry_fee
            position = {
                "direction": direction,
                "entry_time": timestamp.isoformat(),
                "signal_time": str(pending_entry["signal_time"]),
                "entry_price": fill_price,
                "quantity": quantity,
                "notional": notional,
                "margin_used": size_plan.margin_used,
                "entry_fee": entry_fee,
                "stop_price": size_plan.stop_price,
                "take_profit_price": size_plan.take_profit_price,
                "liquidation_price": approximate_liquidation_price(
                    fill_price,
                    direction=direction,
                    leverage=backtest_config.leverage,
                ),
                "bars_held": 0.0,
                "risk_per_unit": size_plan.risk_per_unit,
                "box_id": str(pending_entry["box_id"]),
                "breakeven_armed": False,
                "entry_reason": str(pending_entry["reason"]),
                "box_high": float(pending_entry["box_high"]),
                "box_low": float(pending_entry["box_low"]),
                "stop_family": str(pending_entry["stop_family"]),
                "stop_buffer_atr": float(pending_entry["stop_buffer_atr"]),
            }
            if research_variant.one_trade_per_box:
                used_box_ids.add(str(pending_entry["box_id"]))
            pending_entry = None

        if position is not None:
            exposure_bars += 1
            position["bars_held"] = float(position["bars_held"]) + 1.0

            intrabar_exit = _resolve_intrabar_exit(
                current=current,
                position=position,
                slippage_rate=slippage_rate,
            )
            if intrabar_exit is not None:
                cash, fee_paid = _close_position(
                    cash=cash,
                    position=position,
                    exit_price=intrabar_exit[0],
                    exit_time=timestamp.isoformat(),
                    exit_reason=intrabar_exit[1],
                    trades=trades,
                    fee_rate=exit_fee_rate,
                )
                total_fees_paid += fee_paid
                if intrabar_exit[1] == "Stop-loss":
                    last_stop_exit_at[str(position["direction"])] = timestamp
                position = None
            else:
                _maybe_move_stop_to_breakeven(
                    position=position,
                    current=current,
                    research_variant=research_variant,
                )
                timed_exit = _resolve_time_stop(
                    current=current,
                    position=position,
                    slippage_rate=slippage_rate,
                    research_variant=research_variant,
                )
                if timed_exit is not None:
                    cash, fee_paid = _close_position(
                        cash=cash,
                        position=position,
                        exit_price=timed_exit[0],
                        exit_time=timestamp.isoformat(),
                        exit_reason=timed_exit[1],
                        trades=trades,
                        fee_rate=exit_fee_rate,
                    )
                    total_fees_paid += fee_paid
                    position = None

        if position is None and pending_entry is None and index < len(feature_frame) - 1:
            if entry_start_time is not None and timestamp < entry_start_time:
                marked_equity = cash
                if position is not None:
                    marked_equity += _unrealized_pnl(position=position, mark_price=float(current["close"]))
                equity_rows.append({"timestamp": timestamp.isoformat(), "equity": marked_equity})
                continue
            decision = _entry_decision_from_row(
                current=current,
                strategy_config=strategy_config,
                research_variant=research_variant,
            )
            if (
                decision is not None
                and (entry_gate is None or entry_gate(current, timestamp))
                and _variant_allows_entry(
                    current=current,
                    timestamp=timestamp,
                    decision=decision,
                    research_variant=research_variant,
                    last_stop_exit_at=last_stop_exit_at,
                    used_box_ids=used_box_ids,
                    signal_probabilities=signal_probabilities,
                )
            ):
                pending_entry = {
                    "direction": decision["direction"],
                    "reason": decision["reason"],
                    "stop_reference_price": float(decision["stop_reference_price"]),
                    "reward_risk_ratio": float(decision["reward_risk_ratio"]),
                    "box_id": str(decision["box_id"]),
                    "signal_time": timestamp.isoformat(),
                    "box_high": float(current["box_high"]),
                    "box_low": float(current["box_low"]),
                    "stop_family": str(decision["stop_family"]),
                    "stop_buffer_atr": float(decision["stop_buffer_atr"]),
                }

        marked_equity = cash
        if position is not None:
            marked_equity += _unrealized_pnl(position=position, mark_price=float(current["close"]))
        equity_rows.append({"timestamp": timestamp.isoformat(), "equity": marked_equity})

    if position is not None:
        last_bar = feature_frame.iloc[-1]
        cash, fee_paid = _close_position(
            cash=cash,
            position=position,
            exit_price=float(last_bar["close"]),
            exit_time=feature_frame.index[-1].isoformat(),
            exit_reason="End of backtest",
            trades=trades,
            fee_rate=exit_fee_rate,
        )
        total_fees_paid += fee_paid
        equity_rows.append(
            {
                "timestamp": feature_frame.index[-1].isoformat(),
                "equity": cash,
            }
        )

    trades_frame = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)
    if not equity_curve.empty:
        equity_curve["timestamp"] = pd.to_datetime(equity_curve["timestamp"], utc=True)
        equity_curve = equity_curve.drop_duplicates("timestamp").set_index("timestamp").sort_index()
        if entry_start_time is not None:
            equity_curve = equity_curve[equity_curve.index >= entry_start_time]

    final_equity = cash
    total_return_pct = ((final_equity / backtest_config.starting_cash) - 1) * 100
    max_drawdown_pct = _compute_max_drawdown_pct(equity_curve["equity"]) if not equity_curve.empty else 0.0
    win_rate_pct, profit_factor = _trade_stats(trades_frame)
    exposure_pct = (exposure_bars / max(len(feature_frame) - 1, 1)) * 100

    return AssetBacktestResult(
        product_id=product_id,
        granularity=granularity,
        starting_cash=backtest_config.starting_cash,
        final_equity=final_equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        trades_count=len(trades_frame),
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        exposure_pct=exposure_pct,
        candles_count=len(feature_frame),
        fees_paid=total_fees_paid,
        leverage=backtest_config.leverage,
        trades_frame=trades_frame,
        equity_curve=equity_curve,
    )


def _strategy_with_backtest_leverage(
    strategy_config: BreakoutConfig,
    leverage: float,
) -> BreakoutConfig:
    return BreakoutConfig(
        timeframe_mode=strategy_config.timeframe_mode,
        signal_granularity=strategy_config.signal_granularity,
        context_granularity=strategy_config.context_granularity,
        reward_risk_ratio=strategy_config.reward_risk_ratio,
        atr_period=strategy_config.atr_period,
        volume_window=strategy_config.volume_window,
        min_box_atr_ratio=strategy_config.min_box_atr_ratio,
        min_volume_ratio=strategy_config.min_volume_ratio,
        risk_fraction=strategy_config.risk_fraction,
        leverage=leverage,
        min_position_notional=strategy_config.min_position_notional,
        max_position_notional=strategy_config.max_position_notional,
        kalman_process_variance=strategy_config.kalman_process_variance,
        kalman_measurement_variance=strategy_config.kalman_measurement_variance,
    )


def _fee_rate_for_liquidity(*, maker_rate: float, taker_rate: float, liquidity: str) -> float:
    if liquidity.lower() == "maker":
        return maker_rate
    return taker_rate


def _apply_entry_slippage(price: float, direction: str, slippage_rate: float) -> float:
    if direction == "LONG":
        return price * (1 + slippage_rate)
    return price * (1 - slippage_rate)


def _apply_exit_slippage(price: float, direction: str, slippage_rate: float) -> float:
    if direction == "LONG":
        return price * (1 - slippage_rate)
    return price * (1 + slippage_rate)


def _variant_allows_entry(
    *,
    current: pd.Series,
    timestamp: pd.Timestamp,
    decision: dict[str, float | str],
    research_variant: ResearchVariant,
    last_stop_exit_at: dict[str, pd.Timestamp],
    used_box_ids: set[str],
    signal_probabilities: Optional[dict[str, float]],
) -> bool:
    direction = str(decision["direction"])
    if research_variant.side_mode == SIDE_MODE_LONG_ONLY and direction != "LONG":
        return False
    if research_variant.side_mode == SIDE_MODE_SHORT_ONLY and direction != "SHORT":
        return False
    if timestamp.hour in set(research_variant.blocked_utc_hours):
        return False
    if research_variant.cooldown_minutes > 0:
        last_stop_exit = last_stop_exit_at.get(direction)
        if last_stop_exit is not None:
            cooldown_end = last_stop_exit + pd.Timedelta(minutes=research_variant.cooldown_minutes)
            if timestamp < cooldown_end:
                return False
    if research_variant.one_trade_per_box and str(decision["box_id"]) in used_box_ids:
        return False
    if research_variant.ml_gate_enabled:
        if signal_probabilities is None:
            return False
        probability = float(signal_probabilities.get(timestamp.isoformat(), 0.0))
        if probability < research_variant.ml_gate_threshold:
            return False
    return True


def _entry_decision_from_row(
    *,
    current: pd.Series,
    strategy_config: BreakoutConfig,
    research_variant: ResearchVariant,
) -> Optional[dict[str, float | str]]:
    long_kalman_ok = _directional_kalman_ok(
        current=current,
        direction="LONG",
        threshold=_directional_threshold(
            research_variant.kalman_slope_threshold,
            research_variant.long_kalman_slope_threshold,
        ),
    )
    short_kalman_ok = _directional_kalman_ok(
        current=current,
        direction="SHORT",
        threshold=_directional_threshold(
            research_variant.kalman_slope_threshold,
            research_variant.short_kalman_slope_threshold,
        ),
    )
    long_box_atr_ok = float(current["box_range"]) >= float(current["box_atr"]) * _directional_threshold(
        strategy_config.min_box_atr_ratio,
        research_variant.long_min_box_atr_ratio,
    )
    short_box_atr_ok = float(current["box_range"]) >= float(current["box_atr"]) * _directional_threshold(
        strategy_config.min_box_atr_ratio,
        research_variant.short_min_box_atr_ratio,
    )
    long_volume_ok = float(current["volume"]) >= float(current["volume_ma"]) * _directional_threshold(
        strategy_config.min_volume_ratio,
        research_variant.long_min_volume_ratio,
    )
    short_volume_ok = float(current["volume"]) >= float(current["volume_ma"]) * _directional_threshold(
        strategy_config.min_volume_ratio,
        research_variant.short_min_volume_ratio,
    )
    long_breakout_distance_ok = _breakout_distance_ratio(
        current=current,
        direction="LONG",
    ) >= _directional_threshold(
        research_variant.min_breakout_distance_box_ratio,
        research_variant.long_min_breakout_distance_box_ratio,
    )
    short_breakout_distance_ok = _breakout_distance_ratio(
        current=current,
        direction="SHORT",
    ) >= _directional_threshold(
        research_variant.min_breakout_distance_box_ratio,
        research_variant.short_min_breakout_distance_box_ratio,
    )

    if (
        bool(current["long_breakout"])
        and long_kalman_ok
        and long_box_atr_ok
        and long_volume_ok
        and long_breakout_distance_ok
    ):
        return {
            "direction": "LONG",
            "reason": "Long breakout above prior box high",
            "stop_reference_price": _stop_reference_price(
                current=current,
                direction="LONG",
                research_variant=research_variant,
            ),
            "reward_risk_ratio": _directional_reward_risk_ratio(
                strategy_config.reward_risk_ratio,
                research_variant.long_reward_risk_ratio,
            ),
            "box_id": _box_id_from_row(current),
            "stop_family": research_variant.stop_family,
            "stop_buffer_atr": float(research_variant.stop_buffer_atr),
        }
    if (
        bool(current["short_breakout"])
        and short_kalman_ok
        and short_box_atr_ok
        and short_volume_ok
        and short_breakout_distance_ok
    ):
        return {
            "direction": "SHORT",
            "reason": "Short breakout below prior box low",
            "stop_reference_price": _stop_reference_price(
                current=current,
                direction="SHORT",
                research_variant=research_variant,
            ),
            "reward_risk_ratio": _directional_reward_risk_ratio(
                strategy_config.reward_risk_ratio,
                research_variant.short_reward_risk_ratio,
            ),
            "box_id": _box_id_from_row(current),
            "stop_family": research_variant.stop_family,
            "stop_buffer_atr": float(research_variant.stop_buffer_atr),
        }
    return None


def _directional_threshold(base_value: float, override_value: Optional[float]) -> float:
    return float(override_value if override_value is not None else base_value)


def _directional_reward_risk_ratio(base_value: float, override_value: Optional[float]) -> float:
    return float(override_value if override_value is not None else base_value)


def _breakout_distance_ratio(*, current: pd.Series, direction: str) -> float:
    box_range = max(float(current["box_range"]), 1e-12)
    if direction == "LONG":
        return max(float(current["close"]) - float(current["box_high"]), 0.0) / box_range
    return max(float(current["box_low"]) - float(current["close"]), 0.0) / box_range


def _directional_kalman_ok(
    *,
    current: pd.Series,
    direction: str,
    threshold: float,
) -> bool:
    threshold = max(float(threshold), 0.0)
    if direction == "LONG" and not bool(current["close"] > current["kalman_state"]):
        return False
    if direction == "SHORT" and not bool(current["close"] < current["kalman_state"]):
        return False
    return _normalized_kalman_slope(current=current, direction=direction) >= threshold


def _normalized_kalman_slope(*, current: pd.Series, direction: str) -> float:
    box_atr = max(float(current["box_atr"]), 1e-12)
    normalized = float(current["kalman_slope"]) / box_atr
    if direction == "LONG":
        return normalized
    return -normalized


def _stop_reference_price(
    *,
    current: pd.Series,
    direction: str,
    research_variant: ResearchVariant,
) -> float:
    box_atr = float(current["box_atr"])
    if direction == "LONG":
        candle_extreme = float(current["low"])
        box_edge = float(current["box_low"])
        structural = min(candle_extreme, box_edge)
        if research_variant.stop_family == STOP_FAMILY_BREAKOUT_CANDLE:
            return candle_extreme
        if research_variant.stop_family == STOP_FAMILY_BOX_EDGE:
            return box_edge
        if research_variant.stop_family == STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX:
            return structural
        if research_variant.stop_family == STOP_FAMILY_STRUCTURAL_ATR_BUFFER:
            return structural - box_atr * max(float(research_variant.stop_buffer_atr), 0.0)
    else:
        candle_extreme = float(current["high"])
        box_edge = float(current["box_high"])
        structural = max(candle_extreme, box_edge)
        if research_variant.stop_family == STOP_FAMILY_BREAKOUT_CANDLE:
            return candle_extreme
        if research_variant.stop_family == STOP_FAMILY_BOX_EDGE:
            return box_edge
        if research_variant.stop_family == STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX:
            return structural
        if research_variant.stop_family == STOP_FAMILY_STRUCTURAL_ATR_BUFFER:
            return structural + box_atr * max(float(research_variant.stop_buffer_atr), 0.0)
    raise ValueError(f"Unsupported stop family: {research_variant.stop_family}")


def _box_id_from_row(current: pd.Series) -> str:
    box_start = current.get("box_start")
    if isinstance(box_start, pd.Timestamp):
        return box_start.isoformat()
    return str(box_start)


def _resolve_intrabar_exit(
    *,
    current: pd.Series,
    position: dict[str, float | str],
    slippage_rate: float,
) -> Optional[tuple[float, str]]:
    direction = str(position["direction"])
    low_price = float(current["low"])
    high_price = float(current["high"])
    stop_price = float(position["stop_price"])
    take_profit_price = float(position["take_profit_price"])
    liquidation_price = float(position["liquidation_price"])

    if direction == "LONG":
        if low_price <= liquidation_price:
            return (_apply_exit_slippage(liquidation_price, direction, slippage_rate), "Approx liquidation")
        if low_price <= stop_price:
            return (_apply_exit_slippage(stop_price, direction, slippage_rate), "Stop-loss")
        if high_price >= take_profit_price:
            return (_apply_exit_slippage(take_profit_price, direction, slippage_rate), "Take-profit")
    else:
        if high_price >= liquidation_price:
            return (_apply_exit_slippage(liquidation_price, direction, slippage_rate), "Approx liquidation")
        if high_price >= stop_price:
            return (_apply_exit_slippage(stop_price, direction, slippage_rate), "Stop-loss")
        if low_price <= take_profit_price:
            return (_apply_exit_slippage(take_profit_price, direction, slippage_rate), "Take-profit")
    return None


def _maybe_move_stop_to_breakeven(
    *,
    position: dict[str, float | str | bool],
    current: pd.Series,
    research_variant: ResearchVariant,
) -> None:
    if research_variant.breakeven_trigger_r is None:
        return
    if bool(position.get("breakeven_armed")):
        return
    direction = str(position["direction"])
    entry_price = float(position["entry_price"])
    risk_per_unit = float(position["risk_per_unit"])
    trigger_distance = risk_per_unit * float(research_variant.breakeven_trigger_r)
    if trigger_distance <= 0:
        return
    if direction == "LONG":
        if float(current["high"]) >= entry_price + trigger_distance:
            position["stop_price"] = max(float(position["stop_price"]), entry_price)
            position["breakeven_armed"] = True
    else:
        if float(current["low"]) <= entry_price - trigger_distance:
            position["stop_price"] = min(float(position["stop_price"]), entry_price)
            position["breakeven_armed"] = True


def _resolve_time_stop(
    *,
    current: pd.Series,
    position: dict[str, float | str | bool],
    slippage_rate: float,
    research_variant: ResearchVariant,
) -> Optional[tuple[float, str]]:
    if research_variant.time_stop_bars is None:
        return None
    if int(float(position["bars_held"])) < int(research_variant.time_stop_bars):
        return None
    direction = str(position["direction"])
    entry_price = float(position["entry_price"])
    close_price = float(current["close"])
    if direction == "LONG" and close_price <= entry_price:
        return (_apply_exit_slippage(close_price, direction, slippage_rate), "Time-stop")
    if direction == "SHORT" and close_price >= entry_price:
        return (_apply_exit_slippage(close_price, direction, slippage_rate), "Time-stop")
    return None


def _close_position(
    *,
    cash: float,
    position: dict[str, float | str],
    exit_price: float,
    exit_time: str,
    exit_reason: str,
    trades: list[dict[str, float | str]],
    fee_rate: float,
) -> tuple[float, float]:
    direction = str(position["direction"])
    quantity = float(position["quantity"])
    exit_notional = quantity * exit_price
    if direction == "LONG":
        realized_pnl = quantity * (exit_price - float(position["entry_price"]))
    else:
        realized_pnl = quantity * (float(position["entry_price"]) - exit_price)
    exit_fee = exit_notional * fee_rate
    net_pnl = realized_pnl - float(position["entry_fee"]) - exit_fee
    margin_used = float(position["margin_used"])
    pnl_pct = net_pnl / margin_used * 100 if margin_used else 0.0
    cash += realized_pnl - exit_fee

    trades.append(
        {
            "direction": direction,
            "signal_time": str(position["signal_time"]),
            "entry_time": str(position["entry_time"]),
            "exit_time": exit_time,
            "entry_price": round(float(position["entry_price"]), 8),
            "exit_price": round(exit_price, 8),
            "quantity": round(quantity, 8),
            "notional": round(float(position["notional"]), 8),
            "margin_used": round(margin_used, 8),
            "entry_fee": round(float(position["entry_fee"]), 8),
            "exit_fee": round(exit_fee, 8),
            "pnl": round(net_pnl, 8),
            "pnl_pct": round(pnl_pct, 8),
            "bars_held": int(float(position["bars_held"])),
            "liquidation_price": round(float(position["liquidation_price"]), 8),
            "stop_price": round(float(position["stop_price"]), 8),
            "take_profit_price": round(float(position["take_profit_price"]), 8),
            "box_high": round(float(position["box_high"]), 8),
            "box_low": round(float(position["box_low"]), 8),
            "stop_family": str(position.get("stop_family", STOP_FAMILY_BREAKOUT_CANDLE)),
            "stop_buffer_atr": round(float(position.get("stop_buffer_atr", 0.0)), 8),
            "entry_reason": str(position["entry_reason"]),
            "exit_reason": exit_reason,
        }
    )
    return cash, exit_fee


def _unrealized_pnl(position: dict[str, float | str], mark_price: float) -> float:
    direction = str(position["direction"])
    quantity = float(position["quantity"])
    entry_price = float(position["entry_price"])
    if direction == "LONG":
        return quantity * (mark_price - entry_price)
    return quantity * (entry_price - mark_price)


def _compute_max_drawdown_pct(equity_series: pd.Series) -> float:
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1
    return abs(float(drawdown.min()) * 100)


def _coerce_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _config_with_reward_risk(
    strategy_config: BreakoutConfig,
    reward_risk_ratio: float,
) -> BreakoutConfig:
    return replace(strategy_config, reward_risk_ratio=reward_risk_ratio)


def _trade_stats(trades_frame: pd.DataFrame) -> tuple[float, float]:
    if trades_frame.empty:
        return 0.0, 0.0
    wins = trades_frame[trades_frame["pnl"] > 0]
    losses = trades_frame[trades_frame["pnl"] < 0]
    win_rate_pct = (len(wins) / len(trades_frame)) * 100
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    if gross_loss == 0:
        profit_factor = gross_profit if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    return win_rate_pct, profit_factor
