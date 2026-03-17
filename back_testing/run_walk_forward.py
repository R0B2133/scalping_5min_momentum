from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scalping_5min_momentum.back_testing.engine import (  # noqa: E402
    ResearchVariant,
    SIDE_MODE_BOTH,
    SIDE_MODE_LONG_ONLY,
    SIDE_MODE_SHORT_ONLY,
    STOP_FAMILY_BOX_EDGE,
    STOP_FAMILY_BREAKOUT_CANDLE,
    STOP_FAMILY_STRUCTURAL_ATR_BUFFER,
    STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX,
)
from scalping_5min_momentum.back_testing.optimization import (  # noqa: E402
    DEFAULT_BLOCKED_HOUR_COUNTS,
    DEFAULT_BREAKEVEN_GRID,
    DEFAULT_BREAKOUT_DISTANCE_GRID,
    DEFAULT_CONTEXT_GRANULARITIES,
    DEFAULT_COOLDOWN_GRID,
    DEFAULT_KALMAN_SLOPE_THRESHOLD_GRID,
    DEFAULT_LONG_REWARD_RISK_GRID,
    DEFAULT_MIN_BOX_ATR_GRID,
    DEFAULT_MIN_VOLUME_RATIO_GRID,
    DEFAULT_ONE_TRADE_PER_BOX_OPTIONS,
    DEFAULT_SHORT_REWARD_RISK_GRID,
    DEFAULT_SIDE_MODES,
    DEFAULT_STOP_BUFFER_ATR_GRID,
    DEFAULT_STOP_FAMILIES,
    DEFAULT_TIME_STOP_GRID,
    OptimizationConfig,
    build_base_strategy_config,
    evaluate_variant_walk_forward,
    run_regime_filter_research,
    run_optimization_sequence,
    run_xgboost_filter_research,
)
from scalping_5min_momentum.back_testing.walk_forward import (  # noqa: E402
    default_btc_csv_path,
    generate_walk_forward_windows,
    load_btc_candles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BTC rule-first walk-forward optimization, evaluate one rule variant, or run a separate adaptive-filter research study.",
    )
    parser.add_argument(
        "--mode",
        choices=["optimization_sequence", "single_variant", "regime_filter_research", "xgboost_filter_research"],
        default="optimization_sequence",
    )
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--product-id", default="BTC-PERP")
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--validation-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--starting-cash", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee-rate", type=float, default=0.0002)
    parser.add_argument("--taker-fee-rate", type=float, default=0.0006)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--output-dir", default="output_btc_rule_first_taker_taker")

    parser.add_argument("--context-granularity", default="FIVE_MINUTE")
    parser.add_argument("--context-granularities", nargs="+", default=list(DEFAULT_CONTEXT_GRANULARITIES))
    parser.add_argument("--side-mode", choices=[SIDE_MODE_BOTH, SIDE_MODE_LONG_ONLY, SIDE_MODE_SHORT_ONLY], default=SIDE_MODE_BOTH)
    parser.add_argument("--side-modes", nargs="+", choices=[SIDE_MODE_BOTH, SIDE_MODE_LONG_ONLY, SIDE_MODE_SHORT_ONLY], default=list(DEFAULT_SIDE_MODES))

    parser.add_argument("--blocked-utc-hours", nargs="*", type=int, default=[])
    parser.add_argument("--cooldown-minutes", type=int, default=0)
    parser.add_argument("--one-trade-per-box", action="store_true")
    parser.add_argument("--min-breakout-distance-box-ratio", type=float, default=0.0)
    parser.add_argument("--kalman-slope-threshold", type=float, default=0.0)
    parser.add_argument("--long-kalman-slope-threshold", type=float, default=None)
    parser.add_argument("--short-kalman-slope-threshold", type=float, default=None)
    parser.add_argument("--stop-family", choices=list(DEFAULT_STOP_FAMILIES), default=STOP_FAMILY_BREAKOUT_CANDLE)
    parser.add_argument("--stop-buffer-atr", type=float, default=0.0)
    parser.add_argument("--long-reward-risk-ratio", type=float, default=None)
    parser.add_argument("--short-reward-risk-ratio", type=float, default=None)
    parser.add_argument("--long-min-box-atr-ratio", type=float, default=None)
    parser.add_argument("--short-min-box-atr-ratio", type=float, default=None)
    parser.add_argument("--long-min-volume-ratio", type=float, default=None)
    parser.add_argument("--short-min-volume-ratio", type=float, default=None)
    parser.add_argument("--long-min-breakout-distance-box-ratio", type=float, default=None)
    parser.add_argument("--short-min-breakout-distance-box-ratio", type=float, default=None)
    parser.add_argument("--time-stop-bars", type=int, default=None)
    parser.add_argument("--breakeven-trigger-r", type=float, default=None)

    parser.add_argument("--stop-families", nargs="+", choices=list(DEFAULT_STOP_FAMILIES), default=list(DEFAULT_STOP_FAMILIES))
    parser.add_argument("--stop-buffer-atr-grid", nargs="+", type=float, default=list(DEFAULT_STOP_BUFFER_ATR_GRID))
    parser.add_argument("--long-reward-risk-grid", nargs="+", type=float, default=list(DEFAULT_LONG_REWARD_RISK_GRID))
    parser.add_argument("--short-reward-risk-grid", nargs="+", type=float, default=list(DEFAULT_SHORT_REWARD_RISK_GRID))
    parser.add_argument("--min-box-atr-grid", nargs="+", type=float, default=list(DEFAULT_MIN_BOX_ATR_GRID))
    parser.add_argument("--min-volume-ratio-grid", nargs="+", type=float, default=list(DEFAULT_MIN_VOLUME_RATIO_GRID))
    parser.add_argument("--breakout-distance-grid", nargs="+", type=float, default=list(DEFAULT_BREAKOUT_DISTANCE_GRID))
    parser.add_argument("--kalman-slope-threshold-grid", nargs="+", type=float, default=list(DEFAULT_KALMAN_SLOPE_THRESHOLD_GRID))
    parser.add_argument("--blocked-hour-counts", nargs="+", type=int, default=list(DEFAULT_BLOCKED_HOUR_COUNTS))
    parser.add_argument("--cooldown-grid", nargs="+", type=int, default=list(DEFAULT_COOLDOWN_GRID))
    parser.add_argument("--one-trade-per-box-options", nargs="+", choices=["true", "false"], default=["false", "true"])
    parser.add_argument("--time-stop-grid", nargs="+", type=int, default=[0 if value is None else int(value) for value in DEFAULT_TIME_STOP_GRID])
    parser.add_argument("--breakeven-grid", nargs="+", type=float, default=[0.0 if value is None else float(value) for value in DEFAULT_BREAKEVEN_GRID])
    parser.add_argument("--max-drawdown-limit-pct", type=float, default=30.0)
    parser.add_argument("--min-profit-factor", type=float, default=1.0)
    parser.add_argument("--min-validation-trades-per-fold", type=int, default=10)
    parser.add_argument("--regime-min-samples", type=int, default=8)
    parser.add_argument("--regime-min-profit-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    package_root = Path(__file__).resolve().parent
    csv_path = Path(args.csv_path) if args.csv_path else default_btc_csv_path(package_root)
    output_dir = (package_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candles = load_btc_candles(csv_path)
    optimization_config = OptimizationConfig(
        product_id=args.product_id,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
        starting_cash=args.starting_cash,
        leverage=args.leverage,
        maker_fee_rate=args.maker_fee_rate,
        taker_fee_rate=args.taker_fee_rate,
        slippage_bps=args.slippage_bps,
        max_drawdown_limit_pct=args.max_drawdown_limit_pct,
        min_profit_factor=args.min_profit_factor,
        min_validation_trades_per_fold=args.min_validation_trades_per_fold,
        context_granularities=tuple(args.context_granularities),
        side_modes=tuple(args.side_modes),
        stop_families=tuple(args.stop_families),
        stop_buffer_atr_grid=tuple(float(value) for value in args.stop_buffer_atr_grid),
        long_reward_risk_grid=tuple(float(value) for value in args.long_reward_risk_grid),
        short_reward_risk_grid=tuple(float(value) for value in args.short_reward_risk_grid),
        min_box_atr_grid=tuple(float(value) for value in args.min_box_atr_grid),
        min_volume_ratio_grid=tuple(float(value) for value in args.min_volume_ratio_grid),
        breakout_distance_grid=tuple(float(value) for value in args.breakout_distance_grid),
        kalman_slope_threshold_grid=tuple(float(value) for value in args.kalman_slope_threshold_grid),
        blocked_hour_counts=tuple(int(value) for value in args.blocked_hour_counts),
        cooldown_grid=tuple(int(value) for value in args.cooldown_grid),
        one_trade_per_box_options=tuple(_parse_bool(value) for value in args.one_trade_per_box_options),
        time_stop_grid=tuple(_parse_optional_int(value) for value in args.time_stop_grid),
        breakeven_grid=tuple(_parse_optional_float(value) for value in args.breakeven_grid),
        regime_min_samples=args.regime_min_samples,
        regime_min_profit_factor=args.regime_min_profit_factor,
    )
    windows = generate_walk_forward_windows(
        candles.index,
        train_months=args.train_months,
        validation_months=args.validation_months,
        test_months=args.test_months,
    )

    if args.mode == "optimization_sequence":
        run = run_optimization_sequence(candles, optimization_config)
        _write_dataframe(run.experiment_comparison, output_dir / "experiment_comparison.csv")
        _write_dataframe(run.best_variant_fold_metrics, output_dir / "best_variant_fold_metrics.csv")
        _write_dataframe(run.best_variant_trades, output_dir / "best_variant_trades.csv")
        _write_dataframe(run.best_variant_equity, output_dir / "best_variant_equity.csv")
        _write_dataframe(run.best_variant_monthly_returns, output_dir / "best_variant_monthly_returns.csv")
        summary_payload = dict(run.summary)
        summary_payload["csv_path"] = str(csv_path.resolve())
        summary_payload["output_files"] = {
            "experiment_comparison.csv": str((output_dir / "experiment_comparison.csv").resolve()),
            "best_variant_fold_metrics.csv": str((output_dir / "best_variant_fold_metrics.csv").resolve()),
            "best_variant_trades.csv": str((output_dir / "best_variant_trades.csv").resolve()),
            "best_variant_equity.csv": str((output_dir / "best_variant_equity.csv").resolve()),
            "best_variant_monthly_returns.csv": str((output_dir / "best_variant_monthly_returns.csv").resolve()),
        }
        (output_dir / "optimization_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(json.dumps(summary_payload, indent=2))
        return

    strategy_config = build_base_strategy_config(
        args.leverage,
        context_granularity=args.context_granularity,
    )
    variant = _build_variant_from_args(args)

    if args.mode == "single_variant":
        evaluation = evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=optimization_config,
            strategy_config=strategy_config,
            variant=variant,
            step_name="single_variant",
            windows=windows,
        )
        _write_dataframe(evaluation.fold_metrics, output_dir / "variant_fold_metrics.csv")
        _write_dataframe(evaluation.trades, output_dir / "variant_trades.csv")
        _write_dataframe(evaluation.equity, output_dir / "variant_equity.csv")
        _write_dataframe(evaluation.monthly_returns, output_dir / "variant_monthly_returns.csv")
        summary_payload = {
            "csv_path": str(csv_path.resolve()),
            "variant": evaluation.summary,
            "research_variant": _variant_payload(variant),
            "strategy_config": {
                "signal_granularity": strategy_config.signal_granularity,
                "context_granularity": strategy_config.context_granularity,
            },
            "output_files": {
                "variant_fold_metrics.csv": str((output_dir / "variant_fold_metrics.csv").resolve()),
                "variant_trades.csv": str((output_dir / "variant_trades.csv").resolve()),
                "variant_equity.csv": str((output_dir / "variant_equity.csv").resolve()),
                "variant_monthly_returns.csv": str((output_dir / "variant_monthly_returns.csv").resolve()),
            },
        }
        (output_dir / "variant_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(json.dumps(summary_payload, indent=2))
        return

    if args.mode == "regime_filter_research":
        run = run_regime_filter_research(
            candles,
            optimization_config,
            strategy_config=strategy_config,
            variant=variant,
            windows=windows,
        )
        _write_dataframe(run.fold_metrics, output_dir / "regime_fold_metrics.csv")
        _write_dataframe(run.baseline_trades, output_dir / "baseline_trades.csv")
        _write_dataframe(run.baseline_equity, output_dir / "baseline_equity.csv")
        _write_dataframe(run.baseline_monthly_returns, output_dir / "baseline_monthly_returns.csv")
        _write_dataframe(run.regime_filtered_trades, output_dir / "regime_filtered_trades.csv")
        _write_dataframe(run.regime_filtered_equity, output_dir / "regime_filtered_equity.csv")
        _write_dataframe(run.regime_filtered_monthly_returns, output_dir / "regime_filtered_monthly_returns.csv")
        _write_dataframe(run.train_regime_table, output_dir / "train_regime_table.csv")
        summary_payload = dict(run.summary)
        summary_payload["csv_path"] = str(csv_path.resolve())
        summary_payload["output_files"] = {
            "regime_fold_metrics.csv": str((output_dir / "regime_fold_metrics.csv").resolve()),
            "baseline_trades.csv": str((output_dir / "baseline_trades.csv").resolve()),
            "baseline_equity.csv": str((output_dir / "baseline_equity.csv").resolve()),
            "baseline_monthly_returns.csv": str((output_dir / "baseline_monthly_returns.csv").resolve()),
            "regime_filtered_trades.csv": str((output_dir / "regime_filtered_trades.csv").resolve()),
            "regime_filtered_equity.csv": str((output_dir / "regime_filtered_equity.csv").resolve()),
            "regime_filtered_monthly_returns.csv": str((output_dir / "regime_filtered_monthly_returns.csv").resolve()),
            "train_regime_table.csv": str((output_dir / "train_regime_table.csv").resolve()),
        }
        (output_dir / "regime_filter_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        print(json.dumps(summary_payload, indent=2))
        return

    run = run_xgboost_filter_research(
        candles,
        optimization_config,
        strategy_config=strategy_config,
        variant=variant,
        windows=windows,
    )
    _write_dataframe(run.baseline_fold_metrics, output_dir / "baseline_fold_metrics.csv")
    _write_dataframe(run.baseline_trades, output_dir / "baseline_trades.csv")
    _write_dataframe(run.baseline_equity, output_dir / "baseline_equity.csv")
    _write_dataframe(run.baseline_monthly_returns, output_dir / "baseline_monthly_returns.csv")
    _write_dataframe(run.xgboost_fold_metrics, output_dir / "xgboost_fold_metrics.csv")
    _write_dataframe(run.xgboost_trades, output_dir / "xgboost_trades.csv")
    _write_dataframe(run.xgboost_equity, output_dir / "xgboost_equity.csv")
    _write_dataframe(run.xgboost_monthly_returns, output_dir / "xgboost_monthly_returns.csv")
    summary_payload = dict(run.summary)
    summary_payload["csv_path"] = str(csv_path.resolve())
    summary_payload["output_files"] = {
        "baseline_fold_metrics.csv": str((output_dir / "baseline_fold_metrics.csv").resolve()),
        "baseline_trades.csv": str((output_dir / "baseline_trades.csv").resolve()),
        "baseline_equity.csv": str((output_dir / "baseline_equity.csv").resolve()),
        "baseline_monthly_returns.csv": str((output_dir / "baseline_monthly_returns.csv").resolve()),
        "xgboost_fold_metrics.csv": str((output_dir / "xgboost_fold_metrics.csv").resolve()),
        "xgboost_trades.csv": str((output_dir / "xgboost_trades.csv").resolve()),
        "xgboost_equity.csv": str((output_dir / "xgboost_equity.csv").resolve()),
        "xgboost_monthly_returns.csv": str((output_dir / "xgboost_monthly_returns.csv").resolve()),
    }
    (output_dir / "xgboost_filter_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(json.dumps(summary_payload, indent=2))


def _build_variant_from_args(args: argparse.Namespace) -> ResearchVariant:
    return ResearchVariant(
        name="single_variant",
        side_mode=args.side_mode,
        blocked_utc_hours=tuple(sorted(set(int(hour) for hour in args.blocked_utc_hours))),
        cooldown_minutes=args.cooldown_minutes,
        one_trade_per_box=args.one_trade_per_box,
        min_breakout_distance_box_ratio=args.min_breakout_distance_box_ratio,
        kalman_slope_threshold=args.kalman_slope_threshold,
        long_kalman_slope_threshold=args.long_kalman_slope_threshold,
        short_kalman_slope_threshold=args.short_kalman_slope_threshold,
        stop_family=args.stop_family,
        stop_buffer_atr=args.stop_buffer_atr,
        ml_gate_enabled=False,
        ml_gate_threshold=0.0,
        long_reward_risk_ratio=args.long_reward_risk_ratio,
        short_reward_risk_ratio=args.short_reward_risk_ratio,
        long_min_box_atr_ratio=args.long_min_box_atr_ratio,
        short_min_box_atr_ratio=args.short_min_box_atr_ratio,
        long_min_volume_ratio=args.long_min_volume_ratio,
        short_min_volume_ratio=args.short_min_volume_ratio,
        long_min_breakout_distance_box_ratio=args.long_min_breakout_distance_box_ratio,
        short_min_breakout_distance_box_ratio=args.short_min_breakout_distance_box_ratio,
        time_stop_bars=args.time_stop_bars,
        breakeven_trigger_r=args.breakeven_trigger_r,
    )


def _variant_payload(variant: ResearchVariant) -> dict[str, object]:
    return {
        "name": variant.name,
        "side_mode": variant.side_mode,
        "blocked_utc_hours": list(variant.blocked_utc_hours),
        "cooldown_minutes": variant.cooldown_minutes,
        "one_trade_per_box": variant.one_trade_per_box,
        "min_breakout_distance_box_ratio": variant.min_breakout_distance_box_ratio,
        "kalman_slope_threshold": variant.kalman_slope_threshold,
        "long_kalman_slope_threshold": variant.long_kalman_slope_threshold,
        "short_kalman_slope_threshold": variant.short_kalman_slope_threshold,
        "stop_family": variant.stop_family,
        "stop_buffer_atr": variant.stop_buffer_atr,
        "long_reward_risk_ratio": variant.long_reward_risk_ratio,
        "short_reward_risk_ratio": variant.short_reward_risk_ratio,
        "long_min_box_atr_ratio": variant.long_min_box_atr_ratio,
        "short_min_box_atr_ratio": variant.short_min_box_atr_ratio,
        "long_min_volume_ratio": variant.long_min_volume_ratio,
        "short_min_volume_ratio": variant.short_min_volume_ratio,
        "long_min_breakout_distance_box_ratio": variant.long_min_breakout_distance_box_ratio,
        "short_min_breakout_distance_box_ratio": variant.short_min_breakout_distance_box_ratio,
        "time_stop_bars": variant.time_stop_bars,
        "breakeven_trigger_r": variant.breakeven_trigger_r,
    }


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def _parse_optional_int(value: int) -> int | None:
    return None if int(value) <= 0 else int(value)


def _parse_optional_float(value: float) -> float | None:
    return None if float(value) <= 0 else float(value)


def _write_dataframe(frame, path: Path) -> None:
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    main()
