from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd  # type: ignore

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from scalping_5min_momentum.run_coinbase_scalper import load_state  # noqa: E402
from scalping_5min_momentum.scalping_strategy import (  # noqa: E402
    BreakoutConfig,
    PositionState,
    TIMEFRAME_MODE_FIVE_ONLY,
    TIMEFRAME_MODE_STRICT,
    calculate_position_size,
    evaluate_signal,
)


def _frame_from_rows(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.set_index("timestamp")


class BreakoutStrategyTests(unittest.TestCase):
    def test_strict_mode_emits_long_breakout(self) -> None:
        candles = _frame_from_rows(
            [
                ("2026-01-01T00:00:00Z", 100.0, 100.4, 99.8, 100.2, 100),
                ("2026-01-01T00:01:00Z", 100.2, 100.6, 100.0, 100.4, 105),
                ("2026-01-01T00:02:00Z", 100.4, 100.8, 100.2, 100.6, 110),
                ("2026-01-01T00:03:00Z", 100.6, 101.0, 100.4, 100.8, 115),
                ("2026-01-01T00:04:00Z", 100.8, 101.2, 100.6, 101.0, 120),
                ("2026-01-01T00:05:00Z", 100.7, 100.9, 100.4, 100.6, 90),
                ("2026-01-01T00:06:00Z", 100.6, 100.8, 100.3, 100.5, 88),
                ("2026-01-01T00:07:00Z", 100.5, 100.7, 100.2, 100.4, 86),
                ("2026-01-01T00:08:00Z", 100.4, 100.6, 100.1, 100.3, 84),
                ("2026-01-01T00:09:00Z", 100.3, 100.5, 100.0, 100.2, 82),
                ("2026-01-01T00:10:00Z", 100.3, 100.8, 100.2, 100.7, 130),
                ("2026-01-01T00:11:00Z", 100.7, 101.0, 100.6, 100.9, 135),
                ("2026-01-01T00:12:00Z", 100.9, 101.2, 100.8, 101.1, 140),
                ("2026-01-01T00:13:00Z", 101.1, 101.4, 101.0, 101.3, 150),
                ("2026-01-01T00:14:00Z", 101.3, 101.7, 101.2, 101.6, 160),
            ]
        )
        config = BreakoutConfig(
            timeframe_mode=TIMEFRAME_MODE_STRICT,
            signal_granularity="ONE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=3,
            volume_window=3,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.95,
            reward_risk_ratio=1.5,
        )

        decision = evaluate_signal(candles, config, PositionState())

        self.assertEqual(decision.action, "ENTER_LONG")
        self.assertEqual(decision.direction, "LONG")
        self.assertGreater(decision.reference_price, float(decision.box.high or 0.0))

    def test_five_minute_only_emits_short_breakout(self) -> None:
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
        config = BreakoutConfig(
            timeframe_mode=TIMEFRAME_MODE_FIVE_ONLY,
            signal_granularity="FIVE_MINUTE",
            context_granularity="FIVE_MINUTE",
            atr_period=3,
            volume_window=2,
            min_box_atr_ratio=0.2,
            min_volume_ratio=0.9,
        )

        decision = evaluate_signal(candles, config, PositionState())

        self.assertEqual(decision.action, "ENTER_SHORT")
        self.assertEqual(decision.direction, "SHORT")
        self.assertLess(decision.reference_price, float(decision.box.low or 0.0))

    def test_risk_based_position_size_uses_stop_distance(self) -> None:
        config = BreakoutConfig(
            leverage=5.0,
            risk_fraction=0.01,
            min_position_notional=10.0,
            max_position_notional=2000.0,
            reward_risk_ratio=1.5,
        )

        size_plan = calculate_position_size(
            equity=1000.0,
            entry_price=100.0,
            stop_reference_price=99.0,
            direction="LONG",
            config=config,
            fee_rate=0.0,
        )

        self.assertAlmostEqual(size_plan.quantity, 10.0, places=6)
        self.assertAlmostEqual(size_plan.notional, 1000.0, places=6)
        self.assertAlmostEqual(size_plan.margin_used, 200.0, places=6)
        self.assertAlmostEqual(size_plan.take_profit_price, 101.5, places=6)

    def test_load_state_rejects_legacy_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            state_path = Path(tmp_dir) / "legacy_state.json"
            state_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "product_id": "BTC-PERP-INTX",
                        "position": {},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_state(state_path, "BTC-PERP-INTX", 1000.0)


if __name__ == "__main__":
    unittest.main()
