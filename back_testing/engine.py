from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd  # type: ignore

from scalping_5min_momentum.scalping_strategy import (
    BreakoutConfig,
    approximate_liquidation_price,
    build_signal_frame,
    calculate_position_size,
)


@dataclass
class BacktestConfig:
    starting_cash: float = 10000.0
    leverage: float = 2.0
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0003
    entry_liquidity: str = "taker"
    exit_liquidity: str = "taker"
    slippage_bps: float = 2.0


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
) -> AssetBacktestResult:
    strategy_config = _strategy_with_backtest_leverage(strategy_config, backtest_config.leverage)
    feature_frame = build_signal_frame(candles, strategy_config)
    if len(feature_frame) < 3:
        raise ValueError(f"Not enough candles to backtest {product_id}")

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
                    config=strategy_config,
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
                "entry_reason": str(pending_entry["reason"]),
                "box_high": float(pending_entry["box_high"]),
                "box_low": float(pending_entry["box_low"]),
            }
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
                position = None

        if position is None and pending_entry is None and index < len(feature_frame) - 1:
            decision = _entry_decision_from_row(current=current)
            if decision is not None:
                pending_entry = {
                    "direction": decision["direction"],
                    "reason": decision["reason"],
                    "stop_reference_price": float(decision["stop_reference_price"]),
                    "signal_time": timestamp.isoformat(),
                    "box_high": float(current["box_high"]),
                    "box_low": float(current["box_low"]),
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


def _entry_decision_from_row(current: pd.Series) -> Optional[dict[str, float | str]]:
    if bool(current["long_breakout"]) and bool(current["kalman_long_ok"]) and bool(current["volatility_ok"]) and bool(current["volume_ok"]):
        return {
            "direction": "LONG",
            "reason": "Long breakout above prior box high",
            "stop_reference_price": float(current["low"]),
        }
    if bool(current["short_breakout"]) and bool(current["kalman_short_ok"]) and bool(current["volatility_ok"]) and bool(current["volume_ok"]):
        return {
            "direction": "SHORT",
            "reason": "Short breakout below prior box low",
            "stop_reference_price": float(current["high"]),
        }
    return None


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
