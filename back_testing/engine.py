from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd  # type: ignore

from scalping_5min_momentum.scalping_strategy import ScalpingConfig, add_indicators


@dataclass
class BacktestConfig:
    starting_cash: float = 10000.0
    leverage: float = 50.0
    position_allocation: float = 0.81
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
    position_allocation_pct: float
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
            "position_allocation_pct": round(self.position_allocation_pct, 2),
        }


def run_backtest_for_asset(
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: ScalpingConfig,
    backtest_config: BacktestConfig,
    granularity: str,
) -> AssetBacktestResult:
    feature_frame = add_indicators(candles, strategy_config)
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
    position: Optional[dict[str, float | str]] = None
    pending_order: Optional[dict[str, float | str]] = None
    trades: list[dict[str, float | str]] = []
    equity_rows: list[dict[str, float | str]] = []
    exposure_bars = 0
    total_fees_paid = 0.0

    for index in range(1, len(feature_frame)):
        current = feature_frame.iloc[index]
        previous = feature_frame.iloc[index - 1]
        timestamp = feature_frame.index[index]

        if pending_order is not None:
            pending_action = str(pending_order["action"])
            open_price = float(current["open"])

            if pending_action == "BUY" and position is None:
                fill_price = open_price * (1 + slippage_rate)
                margin_used, notional = _resolve_entry_exposure(
                    equity=cash,
                    position_allocation=backtest_config.position_allocation,
                    leverage=backtest_config.leverage,
                    fee_rate=entry_fee_rate,
                )
                if notional >= strategy_config.min_quote_notional:
                    entry_fee = notional * entry_fee_rate
                    quantity = notional / fill_price
                    cash -= entry_fee
                    total_fees_paid += entry_fee
                    atr = float(pending_order["atr"])
                    position = {
                        "entry_time": timestamp.isoformat(),
                        "entry_price": fill_price,
                        "quantity": quantity,
                        "notional": notional,
                        "margin_used": margin_used,
                        "entry_fee": entry_fee,
                        "stop_price": fill_price - atr * strategy_config.stop_atr_multiple,
                        "take_profit_price": fill_price
                        + atr * strategy_config.take_profit_atr_multiple,
                        "liquidation_price": fill_price
                        * max(0.0, (1 - 1 / max(backtest_config.leverage, 1e-9))),
                        "bars_held": 0.0,
                        "entry_reason": str(pending_order["reason"]),
                    }
            elif pending_action == "SELL" and position is not None:
                cash, trades, fee_paid = _close_position(
                    cash=cash,
                    position=position,
                    exit_price=open_price * (1 - slippage_rate),
                    exit_time=timestamp.isoformat(),
                    exit_reason=str(pending_order["reason"]),
                    trades=trades,
                    fee_rate=exit_fee_rate,
                )
                total_fees_paid += fee_paid
                position = None

            pending_order = None

        if position is not None:
            exposure_bars += 1
            position["bars_held"] = float(position["bars_held"]) + 1.0

            stop_price = float(position["stop_price"])
            take_profit_price = float(position["take_profit_price"])
            liquidation_price = float(position["liquidation_price"])
            low_price = float(current["low"])
            high_price = float(current["high"])

            intrabar_exit: Optional[tuple[float, str]] = None
            if low_price <= liquidation_price:
                intrabar_exit = (
                    liquidation_price * (1 - slippage_rate),
                    "Approx liquidation",
                )
            elif low_price <= stop_price:
                intrabar_exit = (stop_price * (1 - slippage_rate), "ATR stop-loss")
            elif high_price >= take_profit_price:
                intrabar_exit = (take_profit_price * (1 - slippage_rate), "ATR take-profit")

            if intrabar_exit is not None:
                cash, trades, fee_paid = _close_position(
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
            else:
                trailing_stop = max(
                    stop_price,
                    float(current["close"])
                    - float(current["atr"]) * strategy_config.trailing_atr_multiple,
                )
                exit_reason = _close_signal_reason(
                    row=current,
                    trailing_stop=trailing_stop,
                    config=strategy_config,
                )
                if exit_reason and index < len(feature_frame) - 1:
                    pending_order = {"action": "SELL", "reason": exit_reason}
                else:
                    position["stop_price"] = trailing_stop
                    position["take_profit_price"] = take_profit_price

        if position is None and pending_order is None and index < len(feature_frame) - 1:
            checks = _entry_checks(current=current, previous=previous, config=strategy_config)
            if all(checks.values()):
                pending_order = {
                    "action": "BUY",
                    "reason": "EMA trend, RSI, VWAP, and volume aligned",
                    "atr": float(current["atr"]),
                }

        close_price = float(current["close"])
        marked_equity = cash
        if position is not None:
            marked_equity += _unrealized_pnl(position, close_price)
        equity_rows.append({"timestamp": timestamp.isoformat(), "equity": marked_equity})

    if position is not None:
        last_bar = feature_frame.iloc[-1]
        cash, trades, fee_paid = _close_position(
            cash=cash,
            position=position,
            exit_price=float(last_bar["close"]),
            exit_time=feature_frame.index[-1].isoformat(),
            exit_reason="End of backtest",
            trades=trades,
            fee_rate=exit_fee_rate,
        )
        total_fees_paid += fee_paid
        position = None
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
        position_allocation_pct=backtest_config.position_allocation * 100,
        trades_frame=trades_frame,
        equity_curve=equity_curve,
    )


def _resolve_entry_exposure(
    *,
    equity: float,
    position_allocation: float,
    leverage: float,
    fee_rate: float,
) -> tuple[float, float]:
    if equity <= 0 or leverage <= 0 or position_allocation <= 0:
        return 0.0, 0.0
    unclamped_margin = equity * position_allocation
    max_margin = equity / (1 + leverage * fee_rate)
    margin_used = min(unclamped_margin, max_margin)
    notional = margin_used * leverage
    return margin_used, notional


def _fee_rate_for_liquidity(*, maker_rate: float, taker_rate: float, liquidity: str) -> float:
    if liquidity.lower() == "maker":
        return maker_rate
    return taker_rate


def _entry_checks(
    current: pd.Series,
    previous: pd.Series,
    config: ScalpingConfig,
) -> dict[str, bool]:
    close_price = float(current["close"])
    return {
        "trend": close_price > float(current["ema_fast"]) > float(current["ema_slow"]),
        "momentum": config.min_rsi_entry <= float(current["rsi"]) <= config.max_rsi_entry,
        "vwap": close_price >= float(current["vwap"]),
        "volume": float(current["volume"]) >= float(current["volume_ma"]) * config.min_volume_ratio,
        "ema_slope": float(current["ema_fast"]) > float(previous["ema_fast"]),
    }


def _close_signal_reason(
    row: pd.Series,
    trailing_stop: float,
    config: ScalpingConfig,
) -> Optional[str]:
    close_price = float(row["close"])
    if close_price <= trailing_stop:
        return "ATR trailing stop"
    if float(row["rsi"]) <= config.exit_rsi:
        return "RSI momentum fade"
    if close_price < float(row["ema_fast"]) or close_price < float(row["vwap"]):
        return "Intraday trend breakdown"
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
) -> tuple[float, list[dict[str, float | str]], float]:
    quantity = float(position["quantity"])
    exit_notional = quantity * exit_price
    realized_pnl = quantity * (exit_price - float(position["entry_price"]))
    exit_fee = exit_notional * fee_rate
    net_pnl = realized_pnl - float(position["entry_fee"]) - exit_fee
    pnl_pct = net_pnl / float(position["margin_used"]) * 100 if float(position["margin_used"]) else 0.0
    cash += realized_pnl - exit_fee
    trades.append(
        {
            "entry_time": str(position["entry_time"]),
            "exit_time": exit_time,
            "entry_price": round(float(position["entry_price"]), 8),
            "exit_price": round(exit_price, 8),
            "quantity": round(quantity, 8),
            "notional": round(float(position["notional"]), 8),
            "margin_used": round(float(position["margin_used"]), 8),
            "entry_fee": round(float(position["entry_fee"]), 8),
            "exit_fee": round(exit_fee, 8),
            "pnl": round(net_pnl, 8),
            "pnl_pct": round(pnl_pct, 8),
            "bars_held": int(float(position["bars_held"])),
            "liquidation_price": round(float(position["liquidation_price"]), 8),
            "entry_reason": str(position["entry_reason"]),
            "exit_reason": exit_reason,
        }
    )
    return cash, trades, exit_fee


def _unrealized_pnl(position: dict[str, float | str], mark_price: float) -> float:
    quantity = float(position["quantity"])
    return quantity * (mark_price - float(position["entry_price"]))


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
