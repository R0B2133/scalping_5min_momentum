from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd  # type: ignore

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from scalping_5min_momentum.back_testing.engine import (  # noqa: E402
    BacktestConfig,
    ResearchVariant,
    STOP_FAMILY_BOX_EDGE,
    run_backtest_for_asset,
)
from scalping_5min_momentum.scalping_strategy import BreakoutConfig  # noqa: E402


def _frame_from_rows(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")


class BacktestEngineTests(unittest.TestCase):
    def test_backtest_generates_trade_from_breakout(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.7, 100.2, 100),
                ("2026-01-01T00:05:00Z", 100.2, 100.6, 99.9, 100.4, 105),
                ("2026-01-01T00:10:00Z", 100.4, 100.9, 100.1, 100.8, 110),
                ("2026-01-01T00:15:00Z", 100.8, 101.0, 100.2, 100.5, 115),
                ("2026-01-01T00:20:00Z", 100.5, 100.7, 100.0, 100.3, 118),
                ("2026-01-01T00:25:00Z", 100.3, 100.5, 99.9, 100.1, 121),
                ("2026-01-01T00:30:00Z", 100.1, 101.3, 100.0, 101.1, 150),
                ("2026-01-01T00:35:00Z", 101.2, 103.0, 101.0, 102.8, 155),
                ("2026-01-01T00:40:00Z", 102.8, 103.2, 102.4, 103.0, 140),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=2,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
        )

        self.assertEqual(result.trades_count, 3)
        self.assertFalse(result.trades_frame.empty)
        self.assertEqual(result.trades_frame.iloc[0]["exit_reason"], "Stop-loss")
        self.assertEqual(result.trades_frame.iloc[1]["exit_reason"], "Take-profit")
        self.assertGreater(result.final_equity, 1000.0)

    def test_entry_gate_can_block_trades(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.7, 100.2, 100),
                ("2026-01-01T00:05:00Z", 100.2, 100.6, 99.9, 100.4, 105),
                ("2026-01-01T00:10:00Z", 100.4, 100.9, 100.1, 100.8, 110),
                ("2026-01-01T00:15:00Z", 100.8, 101.0, 100.2, 100.5, 115),
                ("2026-01-01T00:20:00Z", 100.5, 100.7, 100.0, 100.3, 118),
                ("2026-01-01T00:25:00Z", 100.3, 100.5, 99.9, 100.1, 121),
                ("2026-01-01T00:30:00Z", 100.1, 101.3, 100.0, 101.1, 150),
                ("2026-01-01T00:35:00Z", 101.2, 103.0, 101.0, 102.8, 155),
                ("2026-01-01T00:40:00Z", 102.8, 103.2, 102.4, 103.0, 140),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=2,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
            entry_gate=lambda _row, _timestamp: False,
        )

        self.assertEqual(result.trades_count, 0)
        self.assertAlmostEqual(result.final_equity, 1000.0, places=6)

    def test_research_variant_long_only_blocks_short_entries(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 102.0, 102.5, 101.8, 102.3, 100),
                ("2026-01-01T00:05:00Z", 102.3, 102.8, 102.0, 102.5, 110),
                ("2026-01-01T00:10:00Z", 102.5, 103.0, 102.2, 102.7, 120),
                ("2026-01-01T00:15:00Z", 102.7, 102.9, 101.8, 102.1, 115),
                ("2026-01-01T00:20:00Z", 102.1, 102.3, 101.6, 101.9, 118),
                ("2026-01-01T00:25:00Z", 101.9, 102.0, 101.3, 101.5, 122),
                ("2026-01-01T00:30:00Z", 101.5, 101.7, 100.8, 101.0, 125),
                ("2026-01-01T00:35:00Z", 101.0, 101.2, 100.3, 100.6, 132),
                ("2026-01-01T00:40:00Z", 100.6, 100.8, 99.9, 100.1, 138),
                ("2026-01-01T00:45:00Z", 100.1, 100.3, 99.2, 99.4, 145),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=3,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
            research_variant=ResearchVariant(name="long_only", side_mode="long_only"),
        )

        self.assertEqual(result.trades_count, 0)
        self.assertAlmostEqual(result.final_equity, 1000.0, places=6)

    def test_blocked_hour_suppression_blocks_entries(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.7, 100.2, 100),
                ("2026-01-01T00:05:00Z", 100.2, 100.6, 99.9, 100.4, 105),
                ("2026-01-01T00:10:00Z", 100.4, 100.9, 100.1, 100.8, 110),
                ("2026-01-01T00:15:00Z", 100.8, 101.0, 100.2, 100.5, 115),
                ("2026-01-01T00:20:00Z", 100.5, 100.7, 100.0, 100.3, 118),
                ("2026-01-01T00:25:00Z", 100.3, 100.5, 99.9, 100.1, 121),
                ("2026-01-01T00:30:00Z", 100.1, 101.3, 100.0, 101.1, 150),
                ("2026-01-01T00:35:00Z", 101.2, 103.0, 101.0, 102.8, 155),
                ("2026-01-01T00:40:00Z", 102.8, 103.2, 102.4, 103.0, 140),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=2,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
            research_variant=ResearchVariant(name="blocked_midnight", blocked_utc_hours=(0,)),
        )

        self.assertEqual(result.trades_count, 0)
        self.assertAlmostEqual(result.final_equity, 1000.0, places=6)

    def test_stop_family_box_edge_uses_box_boundary(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.7, 100.2, 100),
                ("2026-01-01T00:05:00Z", 100.2, 100.6, 99.9, 100.4, 105),
                ("2026-01-01T00:10:00Z", 100.4, 100.9, 100.1, 100.8, 110),
                ("2026-01-01T00:15:00Z", 100.8, 101.0, 100.2, 100.5, 115),
                ("2026-01-01T00:20:00Z", 100.5, 100.7, 100.0, 100.3, 118),
                ("2026-01-01T00:25:00Z", 100.3, 100.5, 99.9, 100.1, 121),
                ("2026-01-01T00:30:00Z", 100.1, 101.3, 100.0, 101.1, 150),
                ("2026-01-01T00:35:00Z", 101.2, 103.0, 101.0, 102.8, 155),
                ("2026-01-01T00:40:00Z", 102.8, 103.2, 102.4, 103.0, 140),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=2,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
            research_variant=ResearchVariant(name="box_edge_stop", stop_family=STOP_FAMILY_BOX_EDGE),
        )

        self.assertFalse(result.trades_frame.empty)
        self.assertAlmostEqual(float(result.trades_frame.iloc[0]["stop_price"]), 99.9, places=6)

    def test_large_kalman_slope_threshold_can_block_entries(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.7, 100.2, 100),
                ("2026-01-01T00:05:00Z", 100.2, 100.6, 99.9, 100.4, 105),
                ("2026-01-01T00:10:00Z", 100.4, 100.9, 100.1, 100.8, 110),
                ("2026-01-01T00:15:00Z", 100.8, 101.0, 100.2, 100.5, 115),
                ("2026-01-01T00:20:00Z", 100.5, 100.7, 100.0, 100.3, 118),
                ("2026-01-01T00:25:00Z", 100.3, 100.5, 99.9, 100.1, 121),
                ("2026-01-01T00:30:00Z", 100.1, 101.3, 100.0, 101.1, 150),
                ("2026-01-01T00:35:00Z", 101.2, 103.0, 101.0, 102.8, 155),
                ("2026-01-01T00:40:00Z", 102.8, 103.2, 102.4, 103.0, 140),
            ]
        )
        strategy_config = BreakoutConfig(
            timeframe_mode="5m_only",
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=2,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
            risk_fraction=0.02,
            leverage=2.0,
            min_position_notional=10.0,
            max_position_notional=2000.0,
        )
        backtest_config = BacktestConfig(
            starting_cash=1000.0,
            leverage=2.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_backtest_for_asset(
            product_id="BTC-PERP",
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity="FIVE_MINUTE",
            research_variant=ResearchVariant(name="slope_gate", kalman_slope_threshold=10.0),
        )

        self.assertEqual(result.trades_count, 0)


if __name__ == "__main__":
    unittest.main()
