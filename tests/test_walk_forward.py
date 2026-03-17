from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd  # type: ignore

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from scalping_5min_momentum.back_testing.data_sources import load_local_coinbase_csv  # noqa: E402
from scalping_5min_momentum.back_testing import optimization as optimization_module  # noqa: E402
from scalping_5min_momentum.back_testing.optimization import (  # noqa: E402
    DEFAULT_CONTEXT_GRANULARITIES,
    OptimizationConfig,
    ResearchVariant,
    WalkForwardWindow as OptimizationWindow,
    evaluate_variant_walk_forward,
    run_regime_filter_research,
    run_optimization_sequence,
    run_xgboost_filter_research,
)
from scalping_5min_momentum.back_testing.walk_forward import (  # noqa: E402
    ML_FEATURE_COLUMNS,
    WalkForwardConfig,
    WalkForwardWindow,
    build_trade_label_frame,
    build_trade_outcome_frame,
    build_variant_candidate_feature_frame,
    generate_walk_forward_windows,
    run_walk_forward_research,
    sample_rule_configs,
)
from scalping_5min_momentum.scalping_strategy import BreakoutConfig  # noqa: E402

try:
    import xgboost  # type: ignore  # noqa: F401

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class WalkForwardTests(unittest.TestCase):
    def test_load_local_coinbase_csv_accepts_download_schema(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp_dir:
            csv_path = Path(tmp_dir) / "btc.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "product_id",
                        "granularity",
                        "timestamp_utc",
                        "start_unix",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                )
                writer.writerow(
                    [
                        "BTC-PERP-INTX",
                        "ONE_MINUTE",
                        "2026-01-01T00:00:00Z",
                        1767225600,
                        100.0,
                        101.0,
                        99.5,
                        100.5,
                        42.0,
                    ]
                )

            frame = load_local_coinbase_csv(csv_path)

            self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
            self.assertEqual(frame.index[0].isoformat(), "2026-01-01T00:00:00+00:00")
            self.assertAlmostEqual(float(frame.iloc[0]["close"]), 100.5, places=6)

    def test_generate_walk_forward_windows_uses_complete_months(self) -> None:
        index = pd.date_range("2023-08-30T04:02:00Z", "2026-03-16T12:22:00Z", freq="D", tz="UTC")

        windows = generate_walk_forward_windows(
            index,
            train_months=6,
            validation_months=1,
            test_months=1,
        )

        self.assertGreater(len(windows), 0)
        self.assertEqual(windows[0].train_start.isoformat(), "2023-09-01T00:00:00+00:00")
        self.assertEqual(windows[0].validation_start.isoformat(), "2024-03-01T00:00:00+00:00")
        self.assertEqual(windows[0].test_start.isoformat(), "2024-04-01T00:00:00+00:00")
        self.assertEqual(windows[-1].test_end.isoformat(), "2026-03-01T00:00:00+00:00")

    def test_sample_rule_configs_is_reproducible_and_includes_default(self) -> None:
        base = BreakoutConfig(
            timeframe_mode="strict_1m_on_5m",
            signal_granularity="ONE_MINUTE",
            context_granularity="FIVE_MINUTE",
            leverage=2.0,
        )

        first = sample_rule_configs(base, max_configs=5, seed=42)
        second = sample_rule_configs(base, max_configs=5, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first[0], base)
        self.assertEqual(len(first), 5)

    def test_build_trade_label_frame_uses_signal_time_join(self) -> None:
        candidate_features = pd.DataFrame(
            [
                {"signal_time": "2026-01-01T00:10:00+00:00", **{column: 1.0 for column in ML_FEATURE_COLUMNS}},
                {"signal_time": "2026-01-01T00:20:00+00:00", **{column: 2.0 for column in ML_FEATURE_COLUMNS}},
            ]
        )
        trades_frame = pd.DataFrame(
            [
                {"signal_time": "2026-01-01T00:10:00+00:00", "pnl": 5.0},
                {"signal_time": "2026-01-01T00:20:00+00:00", "pnl": -2.0},
            ]
        )

        labels = build_trade_label_frame(
            trades_frame=trades_frame,
            candidate_features=candidate_features,
        )

        self.assertEqual(labels["label"].tolist(), [1, 0])
        self.assertEqual(labels["signal_time"].tolist(), trades_frame["signal_time"].tolist())

    def test_build_variant_candidate_feature_frame_respects_variant_thresholds(self) -> None:
        timestamps = pd.to_datetime(
            ["2026-01-01T00:10:00Z", "2026-01-01T00:11:00Z"],
            utc=True,
        )
        feature_frame = pd.DataFrame(
            [
                {
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.8,
                    "close": 100.8,
                    "volume": 120.0,
                    "volume_ma": 100.0,
                    "box_range": 1.0,
                    "box_atr": 1.0,
                    "box_high": 100.5,
                    "box_low": 99.5,
                    "kalman_state": 100.1,
                    "kalman_slope": 0.05,
                    "long_breakout": True,
                    "short_breakout": False,
                    "kalman_long_ok": True,
                    "kalman_short_ok": False,
                    "volatility_ok": True,
                    "volume_ok": True,
                },
                {
                    "open": 100.0,
                    "high": 101.2,
                    "low": 99.8,
                    "close": 100.9,
                    "volume": 180.0,
                    "volume_ma": 100.0,
                    "box_range": 1.0,
                    "box_atr": 1.0,
                    "box_high": 100.5,
                    "box_low": 99.5,
                    "kalman_state": 100.1,
                    "kalman_slope": 0.05,
                    "long_breakout": True,
                    "short_breakout": False,
                    "kalman_long_ok": True,
                    "kalman_short_ok": False,
                    "volatility_ok": True,
                    "volume_ok": True,
                },
            ],
            index=timestamps,
        )
        candidate_features = build_variant_candidate_feature_frame(
            feature_frame,
            strategy_config=BreakoutConfig(
                timeframe_mode="strict_1m_on_5m",
                signal_granularity="ONE_MINUTE",
                context_granularity="FIVE_MINUTE",
                leverage=2.0,
            ),
            research_variant=ResearchVariant(name="strict_volume", long_min_volume_ratio=1.5),
        )

        self.assertEqual(candidate_features["signal_time"].tolist(), [timestamps[1].isoformat()])

    def test_regime_filter_model_allows_positive_expectancy_regimes(self) -> None:
        candidate_features = pd.DataFrame(
            [
                {
                    "signal_time": "2026-01-01T00:10:00+00:00",
                    "direction": "LONG",
                    "direction_long": 1,
                    "box_range_atr_ratio": 1.0,
                    "volume_ratio": 0.8,
                    "breakout_distance_norm": 0.02,
                    "kalman_gap_norm": 0.05,
                    "kalman_slope_norm": 0.02,
                    "candle_range_ratio": 0.5,
                    "candle_body_ratio": 0.3,
                    "utc_hour": 0,
                },
                {
                    "signal_time": "2026-01-01T00:11:00+00:00",
                    "direction": "LONG",
                    "direction_long": 1,
                    "box_range_atr_ratio": 1.0,
                    "volume_ratio": 2.0,
                    "breakout_distance_norm": 0.02,
                    "kalman_gap_norm": 0.05,
                    "kalman_slope_norm": 0.02,
                    "candle_range_ratio": 0.5,
                    "candle_body_ratio": 0.3,
                    "utc_hour": 0,
                },
            ]
        )
        outcome_frame = build_trade_outcome_frame(
            trades_frame=pd.DataFrame(
                [
                    {"signal_time": "2026-01-01T00:10:00+00:00", "pnl": -5.0},
                    {"signal_time": "2026-01-01T00:11:00+00:00", "pnl": 8.0},
                ]
            ),
            candidate_features=candidate_features,
        )

        model, regime_table = optimization_module._fit_regime_filter_model(  # type: ignore[attr-defined]
            candidate_features=candidate_features,
            outcome_frame=outcome_frame,
            min_samples=1,
            min_profit_factor=1.0,
        )
        allowed_signal_times = optimization_module._allowed_signal_times_for_regime_filter(  # type: ignore[attr-defined]
            candidate_features=candidate_features,
            model=model,
        )

        self.assertFalse(regime_table.empty)
        self.assertEqual(allowed_signal_times, {"2026-01-01T00:11:00+00:00"})

    @unittest.skipUnless(HAS_XGBOOST, "xgboost is not installed in the local runtime")
    def test_run_walk_forward_research_smoke(self) -> None:
        candles = _synthetic_minute_candles()
        config = WalkForwardConfig(
            product_id="BTC-PERP",
            train_months=6,
            validation_months=1,
            test_months=1,
            starting_cash=1000.0,
            leverage=2.0,
            taker_fee_rate=0.0,
            slippage_bps=0.0,
            seed=42,
            max_rule_configs=1,
        )
        window = WalkForwardWindow(
            fold_index=1,
            train_start=pd.Timestamp("2026-01-01T03:45:00Z"),
            validation_start=pd.Timestamp("2026-01-01T04:15:00Z"),
            test_start=pd.Timestamp("2026-01-01T04:30:00Z"),
            test_end=pd.Timestamp("2026-01-01T04:45:00Z"),
        )

        result = run_walk_forward_research(candles, config, windows=[window])

        self.assertEqual(len(result.fold_metrics), 1)
        self.assertEqual(len(result.selected_rule_configs), 1)
        self.assertIn("baseline_oos_summary", result.summary)
        self.assertIn("ml_gated_oos_summary", result.summary)

    def test_single_variant_walk_forward_respects_side_mode(self) -> None:
        candles = _synthetic_minute_candles()
        config = OptimizationConfig(
            product_id="BTC-PERP",
            train_months=6,
            validation_months=1,
            test_months=1,
            starting_cash=1000.0,
            leverage=2.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            slippage_bps=0.0,
        )
        evaluation = evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=config,
            strategy_config=BreakoutConfig(
                timeframe_mode="strict_1m_on_5m",
                signal_granularity="ONE_MINUTE",
                context_granularity="FIVE_MINUTE",
                leverage=2.0,
            ),
            variant=ResearchVariant(name="long_only", side_mode="long_only"),
            step_name="single_variant",
            windows=[
                WalkForwardWindow(
                    fold_index=1,
                    train_start=pd.Timestamp("2026-01-01T03:45:00Z"),
                    validation_start=pd.Timestamp("2026-01-01T04:15:00Z"),
                    test_start=pd.Timestamp("2026-01-01T04:30:00Z"),
                    test_end=pd.Timestamp("2026-01-01T04:45:00Z"),
                )
            ],
        )

        self.assertIn("long_trades", evaluation.summary)
        self.assertEqual(int(evaluation.summary["short_trades"]), 0)

    def test_run_optimization_sequence_smoke(self) -> None:
        candles = _synthetic_minute_candles()
        config = OptimizationConfig(
            product_id="BTC-PERP",
            train_months=6,
            validation_months=1,
            test_months=1,
            starting_cash=1000.0,
            leverage=2.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_optimization_sequence(
            candles,
            config,
            windows=[
                OptimizationWindow(
                    fold_index=1,
                    train_start=pd.Timestamp("2026-01-01T03:45:00Z"),
                    validation_start=pd.Timestamp("2026-01-01T04:15:00Z"),
                    test_start=pd.Timestamp("2026-01-01T04:30:00Z"),
                    test_end=pd.Timestamp("2026-01-01T04:45:00Z"),
                )
            ],
        )

        self.assertFalse(result.experiment_comparison.empty)
        self.assertIn("best_variant_summary", result.summary)
        self.assertFalse(result.best_variant_fold_metrics.empty)
        self.assertIn("context_granularity", result.experiment_comparison.columns)
        self.assertIn("stop_family", result.experiment_comparison.columns)
        self.assertIn("long_reward_risk_ratio", result.experiment_comparison.columns)
        self.assertIn("long_min_volume_ratio", result.experiment_comparison.columns)
        self.assertNotIn("ml_hard_gate", result.experiment_comparison["step_name"].tolist())

    def test_acceptance_assessment_rejects_low_validation_trade_counts(self) -> None:
        evaluation = optimization_module.VariantEvaluation(
            step_name="entry_quality_tightening",
            variant=ResearchVariant(name="candidate"),
            strategy_config=BreakoutConfig(
                timeframe_mode="strict_1m_on_5m",
                signal_granularity="ONE_MINUTE",
                context_granularity="FIVE_MINUTE",
                leverage=2.0,
            ),
            summary={
                "total_return_pct": 5.0,
                "max_drawdown_pct": 10.0,
                "profit_factor": 1.2,
                "trades_count": 20,
                "fees_paid": 0.0,
                "long_trades": 10,
                "long_win_rate_pct": 50.0,
                "long_net_pnl": 5.0,
                "short_trades": 10,
                "short_win_rate_pct": 50.0,
                "short_net_pnl": 5.0,
            },
            fold_metrics=pd.DataFrame([{"validation_trades_count": 5}]),
            trades=pd.DataFrame(),
            equity=pd.DataFrame(),
            monthly_returns=pd.DataFrame(),
            entry_liquidity="taker",
            exit_liquidity="taker",
            selected_thresholds=[],
        )
        accepted, reason = optimization_module._acceptance_assessment(  # type: ignore[attr-defined]
            evaluation,
            OptimizationConfig(min_validation_trades_per_fold=10),
        )

        self.assertFalse(accepted)
        self.assertIn("Validation trades", reason)

    def test_xgboost_filter_research_runs_as_separate_mode(self) -> None:
        candles = _synthetic_minute_candles()
        config = OptimizationConfig(
            product_id="BTC-PERP",
            train_months=6,
            validation_months=1,
            test_months=1,
            starting_cash=1000.0,
            leverage=2.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            slippage_bps=0.0,
        )

        result = run_xgboost_filter_research(
            candles,
            config,
            strategy_config=BreakoutConfig(
                timeframe_mode="strict_1m_on_5m",
                signal_granularity="ONE_MINUTE",
                context_granularity=DEFAULT_CONTEXT_GRANULARITIES[0],
                leverage=2.0,
            ),
            variant=ResearchVariant(name="candidate"),
            windows=[
                OptimizationWindow(
                    fold_index=1,
                    train_start=pd.Timestamp("2026-01-01T03:45:00Z"),
                    validation_start=pd.Timestamp("2026-01-01T04:15:00Z"),
                    test_start=pd.Timestamp("2026-01-01T04:30:00Z"),
                    test_end=pd.Timestamp("2026-01-01T04:45:00Z"),
                )
            ],
        )

        self.assertIn("baseline_summary", result.summary)
        self.assertIn("xgboost_summary", result.summary)

    def test_regime_filter_research_runs_as_separate_mode(self) -> None:
        candles = _synthetic_minute_candles()
        config = OptimizationConfig(
            product_id="BTC-PERP",
            train_months=6,
            validation_months=1,
            test_months=1,
            starting_cash=1000.0,
            leverage=2.0,
            taker_fee_rate=0.0,
            maker_fee_rate=0.0,
            slippage_bps=0.0,
            regime_min_samples=1,
            regime_min_profit_factor=0.0,
        )

        result = run_regime_filter_research(
            candles,
            config,
            strategy_config=BreakoutConfig(
                timeframe_mode="strict_1m_on_5m",
                signal_granularity="ONE_MINUTE",
                context_granularity=DEFAULT_CONTEXT_GRANULARITIES[0],
                leverage=2.0,
            ),
            variant=ResearchVariant(name="candidate"),
            windows=[
                OptimizationWindow(
                    fold_index=1,
                    train_start=pd.Timestamp("2026-01-01T03:45:00Z"),
                    validation_start=pd.Timestamp("2026-01-01T04:15:00Z"),
                    test_start=pd.Timestamp("2026-01-01T04:30:00Z"),
                    test_end=pd.Timestamp("2026-01-01T04:45:00Z"),
                )
            ],
        )

        self.assertIn("baseline_summary", result.summary)
        self.assertIn("regime_filtered_summary", result.summary)
        self.assertFalse(result.fold_metrics.empty)
        self.assertIn("allowed_regimes_count", result.fold_metrics.columns)


def _synthetic_minute_candles() -> pd.DataFrame:
    long_pattern = [
        (100.0, 100.4, 99.8, 100.2, 100),
        (100.2, 100.6, 100.0, 100.4, 105),
        (100.4, 100.8, 100.2, 100.6, 110),
        (100.6, 101.0, 100.4, 100.8, 115),
        (100.8, 101.2, 100.6, 101.0, 120),
        (100.7, 100.9, 100.4, 100.6, 90),
        (100.6, 100.8, 100.3, 100.5, 88),
        (100.5, 100.7, 100.2, 100.4, 86),
        (100.4, 100.6, 100.1, 100.3, 84),
        (100.3, 100.5, 100.0, 100.2, 82),
        (100.3, 100.8, 100.2, 100.7, 130),
        (100.7, 101.0, 100.6, 100.9, 135),
        (100.9, 101.2, 100.8, 101.1, 140),
        (101.1, 101.4, 101.0, 101.3, 150),
        (101.3, 101.7, 101.2, 101.6, 160),
    ]
    short_pattern = [
        (102.0, 102.5, 101.8, 102.3, 100),
        (102.3, 102.8, 102.0, 102.5, 110),
        (102.5, 103.0, 102.2, 102.7, 120),
        (102.7, 102.9, 101.8, 102.1, 115),
        (102.1, 102.3, 101.6, 101.9, 118),
        (101.9, 102.0, 101.3, 101.5, 122),
        (101.5, 101.7, 100.8, 101.0, 125),
        (101.0, 101.2, 100.3, 100.6, 132),
        (100.6, 100.8, 99.9, 100.1, 138),
        (100.1, 100.3, 99.2, 99.4, 145),
        (99.5, 99.7, 98.9, 99.1, 155),
        (99.1, 99.3, 98.5, 98.8, 160),
        (98.8, 99.0, 98.2, 98.5, 162),
        (98.5, 98.8, 98.0, 98.2, 165),
        (98.2, 98.6, 97.8, 97.9, 170),
    ]
    rows = []
    timestamp = pd.Timestamp("2026-01-01T00:00:00Z")
    for cycle_index in range(20):
        pattern = long_pattern if cycle_index % 2 == 0 else short_pattern
        for open_price, high_price, low_price, close_price, volume in pattern:
            rows.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )
            timestamp += pd.Timedelta(minutes=1)
    frame = pd.DataFrame(rows)
    return frame.set_index("timestamp")


if __name__ == "__main__":
    unittest.main()
