from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any, Callable, Optional

import pandas as pd  # type: ignore

from scalping_5min_momentum.back_testing.engine import (
    AssetBacktestResult,
    BacktestConfig,
    ResearchVariant,
    SIDE_MODE_BOTH,
    SIDE_MODE_LONG_ONLY,
    SIDE_MODE_SHORT_ONLY,
    STOP_FAMILY_BOX_EDGE,
    STOP_FAMILY_BREAKOUT_CANDLE,
    STOP_FAMILY_STRUCTURAL_ATR_BUFFER,
    STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX,
    run_backtest_for_asset,
)
from scalping_5min_momentum.back_testing.walk_forward import (
    THRESHOLD_GRID,
    build_candidate_feature_frame,
    build_trade_outcome_frame,
    build_variant_candidate_feature_frame,
    build_trade_label_frame,
    fit_xgboost_model,
    generate_walk_forward_windows,
    monthly_returns_from_equity,
    predict_candidate_probabilities,
)
from scalping_5min_momentum.coinbase_advanced import GRANULARITY_TO_SECONDS
from scalping_5min_momentum.scalping_strategy import (
    BreakoutConfig,
    TIMEFRAME_MODE_STRICT,
    build_signal_frame,
    normalize_config,
    required_history_bars,
)

DEFAULT_CONTEXT_GRANULARITIES = ("FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE")
DEFAULT_SIDE_MODES = (SIDE_MODE_BOTH, SIDE_MODE_LONG_ONLY, SIDE_MODE_SHORT_ONLY)
DEFAULT_STOP_FAMILIES = (
    STOP_FAMILY_BREAKOUT_CANDLE,
    STOP_FAMILY_BOX_EDGE,
    STOP_FAMILY_WORSE_OF_CANDLE_AND_BOX,
    STOP_FAMILY_STRUCTURAL_ATR_BUFFER,
)
DEFAULT_STOP_BUFFER_ATR_GRID = (0.1, 0.2)
DEFAULT_LONG_REWARD_RISK_GRID = (1.25, 1.5, 1.75, 2.0, 2.5)
DEFAULT_SHORT_REWARD_RISK_GRID = (1.0, 1.25, 1.5, 1.75)
DEFAULT_MIN_BOX_ATR_GRID = (0.8, 1.0, 1.2, 1.5)
DEFAULT_MIN_VOLUME_RATIO_GRID = (1.0, 1.1, 1.2, 1.4)
DEFAULT_BREAKOUT_DISTANCE_GRID = (0.0, 0.01, 0.02, 0.05)
DEFAULT_KALMAN_SLOPE_THRESHOLD_GRID = (0.0, 0.01, 0.02)
DEFAULT_BLOCKED_HOUR_COUNTS = (4, 6, 8)
DEFAULT_COOLDOWN_GRID = (3, 5, 10)
DEFAULT_ONE_TRADE_PER_BOX_OPTIONS = (False, True)
DEFAULT_TIME_STOP_GRID = (None, 3, 5, 10)
DEFAULT_BREAKEVEN_GRID = (None, 0.5, 1.0)
REGIME_BUCKET_LABELS = ("low", "mid", "high")


@dataclass(frozen=True)
class OptimizationConfig:
    product_id: str = "BTC-PERP"
    train_months: int = 6
    validation_months: int = 1
    test_months: int = 1
    starting_cash: float = 10000.0
    leverage: float = 2.0
    taker_fee_rate: float = 0.0006
    maker_fee_rate: float = 0.0002
    slippage_bps: float = 2.0
    max_drawdown_limit_pct: float = 30.0
    min_profit_factor: float = 1.0
    min_validation_trades_per_fold: int = 10
    context_granularities: tuple[str, ...] = DEFAULT_CONTEXT_GRANULARITIES
    side_modes: tuple[str, ...] = DEFAULT_SIDE_MODES
    stop_families: tuple[str, ...] = DEFAULT_STOP_FAMILIES
    stop_buffer_atr_grid: tuple[float, ...] = DEFAULT_STOP_BUFFER_ATR_GRID
    long_reward_risk_grid: tuple[float, ...] = DEFAULT_LONG_REWARD_RISK_GRID
    short_reward_risk_grid: tuple[float, ...] = DEFAULT_SHORT_REWARD_RISK_GRID
    min_box_atr_grid: tuple[float, ...] = DEFAULT_MIN_BOX_ATR_GRID
    min_volume_ratio_grid: tuple[float, ...] = DEFAULT_MIN_VOLUME_RATIO_GRID
    breakout_distance_grid: tuple[float, ...] = DEFAULT_BREAKOUT_DISTANCE_GRID
    kalman_slope_threshold_grid: tuple[float, ...] = DEFAULT_KALMAN_SLOPE_THRESHOLD_GRID
    blocked_hour_counts: tuple[int, ...] = DEFAULT_BLOCKED_HOUR_COUNTS
    cooldown_grid: tuple[int, ...] = DEFAULT_COOLDOWN_GRID
    one_trade_per_box_options: tuple[bool, ...] = DEFAULT_ONE_TRADE_PER_BOX_OPTIONS
    time_stop_grid: tuple[Optional[int], ...] = DEFAULT_TIME_STOP_GRID
    breakeven_grid: tuple[Optional[float], ...] = DEFAULT_BREAKEVEN_GRID
    regime_min_samples: int = 8
    regime_min_profit_factor: float = 1.0


@dataclass(frozen=True)
class WalkForwardWindow:
    fold_index: int
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_end(self) -> pd.Timestamp:
        return self.validation_start

    @property
    def validation_end(self) -> pd.Timestamp:
        return self.test_start


@dataclass(frozen=True)
class EvaluationCandidate:
    strategy_config: BreakoutConfig
    variant: ResearchVariant


@dataclass
class VariantEvaluation:
    step_name: str
    variant: ResearchVariant
    strategy_config: BreakoutConfig
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    monthly_returns: pd.DataFrame
    entry_liquidity: str
    exit_liquidity: str
    selected_thresholds: list[float]
    status: str = "ok"
    note: str = ""


@dataclass
class OptimizationRun:
    summary: dict[str, Any]
    experiment_comparison: pd.DataFrame
    best_variant_fold_metrics: pd.DataFrame
    best_variant_trades: pd.DataFrame
    best_variant_equity: pd.DataFrame
    best_variant_monthly_returns: pd.DataFrame


@dataclass
class XGBoostFilterRun:
    summary: dict[str, Any]
    baseline_fold_metrics: pd.DataFrame
    baseline_trades: pd.DataFrame
    baseline_equity: pd.DataFrame
    baseline_monthly_returns: pd.DataFrame
    xgboost_fold_metrics: pd.DataFrame
    xgboost_trades: pd.DataFrame
    xgboost_equity: pd.DataFrame
    xgboost_monthly_returns: pd.DataFrame


@dataclass
class RegimeFilterRun:
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    baseline_trades: pd.DataFrame
    baseline_equity: pd.DataFrame
    baseline_monthly_returns: pd.DataFrame
    regime_filtered_trades: pd.DataFrame
    regime_filtered_equity: pd.DataFrame
    regime_filtered_monthly_returns: pd.DataFrame
    train_regime_table: pd.DataFrame


@dataclass
class _WindowBacktest:
    result: AssetBacktestResult
    feature_frame: pd.DataFrame


@dataclass(frozen=True)
class RegimeFilterModel:
    volume_edges: tuple[float, float]
    breakout_edges: tuple[float, float]
    box_range_edges: tuple[float, float]
    allowed_regimes: frozenset[tuple[str, str, str, str]]


def build_base_strategy_config(
    leverage: float,
    *,
    context_granularity: str = "FIVE_MINUTE",
) -> BreakoutConfig:
    return normalize_config(
        BreakoutConfig(
            timeframe_mode=TIMEFRAME_MODE_STRICT,
            signal_granularity="ONE_MINUTE",
            context_granularity=context_granularity,
            leverage=leverage,
        )
    )


def run_optimization_sequence(
    candles: pd.DataFrame,
    config: OptimizationConfig,
    *,
    windows: Optional[list[WalkForwardWindow]] = None,
) -> OptimizationRun:
    baseline_strategy = build_base_strategy_config(
        config.leverage,
        context_granularity=_default_context_granularity(config),
    )
    windows = windows or _convert_windows(
        generate_walk_forward_windows(
            candles.index,
            train_months=config.train_months,
            validation_months=config.validation_months,
            test_months=config.test_months,
        )
    )
    if not windows:
        raise ValueError("No walk-forward windows available for the BTC optimization sequence")

    comparison_rows: list[dict[str, Any]] = []
    step_decisions: list[dict[str, Any]] = []

    baseline = evaluate_variant_walk_forward(
        candles=candles,
        optimization_config=config,
        strategy_config=baseline_strategy,
        variant=ResearchVariant(name="baseline"),
        step_name="baseline",
        windows=windows,
    )
    current_best = baseline
    comparison_rows.append(
        _comparison_row(
            reference=baseline,
            candidate=baseline,
            promoted=True,
            acceptance=_acceptance_assessment(baseline, config),
        )
    )
    step_decisions.append(
        {
            "step_name": "baseline",
            "selected_variant": baseline.variant.name,
            "selected_context_granularity": baseline.strategy_config.context_granularity,
            "promoted": True,
            "reason": "Initial rule-based reference",
        }
    )

    for step_name, candidates in (
        ("side_selection", _side_selection_candidates(current_best, config)),
        ("context_timeframe_sweep", _context_timeframe_candidates(current_best, config)),
        ("trade_suppression", _trade_suppression_candidates(current_best, config)),
        ("entry_quality_tightening", _entry_quality_candidates(current_best, config)),
        ("stop_target_redesign", _exit_redesign_candidates(current_best, config)),
    ):
        current_best, step_rows = _run_step(
            step_name=step_name,
            reference=current_best,
            candidates=candidates,
            candles=candles,
            config=config,
            windows=windows,
        )
        comparison_rows.extend(step_rows[0])
        step_decisions.append(step_rows[1])

    sensitivity_evaluations = _execution_sensitivity_evaluations(
        current_best=current_best,
        candles=candles,
        config=config,
        windows=windows,
    )
    for evaluation in sensitivity_evaluations:
        comparison_rows.append(
            _comparison_row(
                reference=current_best,
                candidate=evaluation,
                promoted=False,
                acceptance=_acceptance_assessment(evaluation, config),
            )
        )
    step_decisions.append(
        {
            "step_name": "execution_sensitivity",
            "selected_variant": current_best.variant.name,
            "selected_context_granularity": current_best.strategy_config.context_granularity,
            "promoted": False,
            "reason": "Validation-only step; execution variants are reported but not promoted",
        }
    )

    summary = {
        "product_id": config.product_id,
        "walk_forward": asdict(config),
        "acceptance_criteria": {
            "maximize": "cumulative_out_of_sample_return_pct",
            "max_drawdown_limit_pct": config.max_drawdown_limit_pct,
            "min_profit_factor": config.min_profit_factor,
            "min_validation_trades_per_fold": config.min_validation_trades_per_fold,
            "fees_model": "taker_taker",
            "slippage_bps": config.slippage_bps,
        },
        "baseline_summary": baseline.summary,
        "best_variant_summary": current_best.summary,
        "final_selected_variant": _variant_to_dict(current_best.variant),
        "final_selected_strategy_config": _strategy_config_to_dict(current_best.strategy_config),
        "step_decisions": step_decisions,
        "xgboost_phase_available": True,
        "xgboost_phase_default_enabled": False,
    }
    return OptimizationRun(
        summary=summary,
        experiment_comparison=pd.DataFrame(comparison_rows),
        best_variant_fold_metrics=current_best.fold_metrics,
        best_variant_trades=current_best.trades,
        best_variant_equity=current_best.equity,
        best_variant_monthly_returns=current_best.monthly_returns,
    )


def run_xgboost_filter_research(
    candles: pd.DataFrame,
    config: OptimizationConfig,
    *,
    strategy_config: BreakoutConfig,
    variant: ResearchVariant,
    windows: Optional[list[WalkForwardWindow]] = None,
) -> XGBoostFilterRun:
    windows = windows or _convert_windows(
        generate_walk_forward_windows(
            candles.index,
            train_months=config.train_months,
            validation_months=config.validation_months,
            test_months=config.test_months,
        )
    )
    if not windows:
        raise ValueError("No walk-forward windows available for XGBoost filter research")

    baseline_variant = replace(
        variant,
        name=f"{variant.name}_rule_baseline",
        ml_gate_enabled=False,
        ml_gate_threshold=0.0,
    )
    filtered_variant = replace(
        variant,
        name=f"{variant.name}_xgboost_filter",
        ml_gate_enabled=True,
        ml_gate_threshold=0.0,
    )
    baseline = evaluate_variant_walk_forward(
        candles=candles,
        optimization_config=config,
        strategy_config=strategy_config,
        variant=baseline_variant,
        step_name="rule_baseline",
        windows=windows,
    )
    filtered = evaluate_variant_walk_forward(
        candles=candles,
        optimization_config=config,
        strategy_config=strategy_config,
        variant=filtered_variant,
        step_name="xgboost_filter_research",
        windows=windows,
    )
    summary = {
        "product_id": config.product_id,
        "walk_forward": asdict(config),
        "strategy_config": _strategy_config_to_dict(strategy_config),
        "base_variant": _variant_to_dict(baseline_variant),
        "baseline_summary": baseline.summary,
        "xgboost_summary": filtered.summary,
    }
    return XGBoostFilterRun(
        summary=summary,
        baseline_fold_metrics=baseline.fold_metrics,
        baseline_trades=baseline.trades,
        baseline_equity=baseline.equity,
        baseline_monthly_returns=baseline.monthly_returns,
        xgboost_fold_metrics=filtered.fold_metrics,
        xgboost_trades=filtered.trades,
        xgboost_equity=filtered.equity,
        xgboost_monthly_returns=filtered.monthly_returns,
    )


def run_regime_filter_research(
    candles: pd.DataFrame,
    config: OptimizationConfig,
    *,
    strategy_config: BreakoutConfig,
    variant: ResearchVariant,
    windows: Optional[list[WalkForwardWindow]] = None,
) -> RegimeFilterRun:
    windows = windows or _convert_windows(
        generate_walk_forward_windows(
            candles.index,
            train_months=config.train_months,
            validation_months=config.validation_months,
            test_months=config.test_months,
        )
    )
    if not windows:
        raise ValueError("No walk-forward windows available for adaptive regime-filter research")

    base_variant = replace(variant, name=f"{variant.name}_rule_baseline", ml_gate_enabled=False, ml_gate_threshold=0.0)
    filtered_variant = replace(variant, name=f"{variant.name}_regime_filter", ml_gate_enabled=False, ml_gate_threshold=0.0)
    baseline_cash = config.starting_cash
    filtered_cash = config.starting_cash
    fold_rows: list[dict[str, Any]] = []
    baseline_trade_frames: list[pd.DataFrame] = []
    baseline_equity_frames: list[pd.DataFrame] = []
    filtered_trade_frames: list[pd.DataFrame] = []
    filtered_equity_frames: list[pd.DataFrame] = []
    regime_rows: list[dict[str, Any]] = []

    for window in windows:
        shared_backtest = _build_backtest_config(
            starting_cash=config.starting_cash,
            leverage=config.leverage,
            maker_fee_rate=config.maker_fee_rate,
            taker_fee_rate=config.taker_fee_rate,
            slippage_bps=config.slippage_bps,
            entry_liquidity="taker",
            exit_liquidity="taker",
        )
        train_backtest = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest,
            window_start=window.train_start,
            window_end=window.train_end,
            research_variant=base_variant,
        )
        validation_baseline = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest,
            window_start=window.validation_start,
            window_end=window.validation_end,
            research_variant=base_variant,
        )

        train_candidates = build_variant_candidate_feature_frame(
            train_backtest.feature_frame,
            strategy_config=strategy_config,
            research_variant=base_variant,
        )
        train_outcomes = build_trade_outcome_frame(
            trades_frame=train_backtest.result.trades_frame,
            candidate_features=train_candidates,
        )
        regime_model, train_regime_table = _fit_regime_filter_model(
            candidate_features=train_candidates,
            outcome_frame=train_outcomes,
            min_samples=config.regime_min_samples,
            min_profit_factor=config.regime_min_profit_factor,
        )
        if not train_regime_table.empty:
            attached_regimes = train_regime_table.copy()
            attached_regimes.insert(0, "fold_index", window.fold_index)
            regime_rows.extend(attached_regimes.to_dict("records"))

        validation_candidates = build_variant_candidate_feature_frame(
            validation_baseline.feature_frame,
            strategy_config=strategy_config,
            research_variant=base_variant,
        )
        validation_allowed_times = _allowed_signal_times_for_regime_filter(
            candidate_features=validation_candidates,
            model=regime_model,
        )
        validation_filtered = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest,
            window_start=window.validation_start,
            window_end=window.validation_end,
            research_variant=filtered_variant,
            entry_gate=_entry_gate_from_signal_times(validation_allowed_times),
        )

        baseline_test = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=_build_backtest_config(
                starting_cash=baseline_cash,
                leverage=config.leverage,
                maker_fee_rate=config.maker_fee_rate,
                taker_fee_rate=config.taker_fee_rate,
                slippage_bps=config.slippage_bps,
                entry_liquidity="taker",
                exit_liquidity="taker",
            ),
            window_start=window.test_start,
            window_end=window.test_end,
            research_variant=base_variant,
        )
        test_candidates = build_variant_candidate_feature_frame(
            _window_backtest_feature_frame(
                candles=candles,
                strategy_config=strategy_config,
                window_start=window.test_start,
                window_end=window.test_end,
            ),
            strategy_config=strategy_config,
            research_variant=base_variant,
        )
        test_allowed_times = _allowed_signal_times_for_regime_filter(
            candidate_features=test_candidates,
            model=regime_model,
        )
        filtered_test = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=_build_backtest_config(
                starting_cash=filtered_cash,
                leverage=config.leverage,
                maker_fee_rate=config.maker_fee_rate,
                taker_fee_rate=config.taker_fee_rate,
                slippage_bps=config.slippage_bps,
                entry_liquidity="taker",
                exit_liquidity="taker",
            ),
            window_start=window.test_start,
            window_end=window.test_end,
            research_variant=filtered_variant,
            entry_gate=_entry_gate_from_signal_times(test_allowed_times),
        )

        baseline_cash = baseline_test.result.final_equity
        filtered_cash = filtered_test.result.final_equity
        baseline_trade_frames.append(_attach_fold_column(baseline_test.result.trades_frame, window.fold_index))
        baseline_equity_frames.append(_attach_fold_column(baseline_test.result.equity_curve.reset_index(), window.fold_index))
        filtered_trade_frames.append(_attach_fold_column(filtered_test.result.trades_frame, window.fold_index))
        filtered_equity_frames.append(_attach_fold_column(filtered_test.result.equity_curve.reset_index(), window.fold_index))
        fold_rows.append(
            {
                "fold_index": window.fold_index,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "train_candidate_count": int(len(train_candidates)),
                "train_labeled_trades": int(len(train_outcomes)),
                "learned_regimes_count": int(len(train_regime_table)),
                "allowed_regimes_count": int(train_regime_table["allowed"].sum()) if not train_regime_table.empty else 0,
                "validation_baseline_return_pct": round(validation_baseline.result.total_return_pct, 8),
                "validation_baseline_profit_factor": round(validation_baseline.result.profit_factor, 8),
                "validation_baseline_trades_count": int(validation_baseline.result.trades_count),
                "validation_filtered_return_pct": round(validation_filtered.result.total_return_pct, 8),
                "validation_filtered_profit_factor": round(validation_filtered.result.profit_factor, 8),
                "validation_filtered_trades_count": int(validation_filtered.result.trades_count),
                "test_baseline_return_pct": round(baseline_test.result.total_return_pct, 8),
                "test_baseline_profit_factor": round(baseline_test.result.profit_factor, 8),
                "test_baseline_trades_count": int(baseline_test.result.trades_count),
                "test_filtered_return_pct": round(filtered_test.result.total_return_pct, 8),
                "test_filtered_profit_factor": round(filtered_test.result.profit_factor, 8),
                "test_filtered_trades_count": int(filtered_test.result.trades_count),
                "baseline_test_final_equity": round(baseline_test.result.final_equity, 8),
                "filtered_test_final_equity": round(filtered_test.result.final_equity, 8),
            }
        )

    baseline_trades = _concat_frames(baseline_trade_frames)
    baseline_equity = _concat_frames(baseline_equity_frames)
    filtered_trades = _concat_frames(filtered_trade_frames)
    filtered_equity = _concat_frames(filtered_equity_frames)
    baseline_monthly_returns = monthly_returns_from_equity(baseline_equity)
    filtered_monthly_returns = monthly_returns_from_equity(filtered_equity)
    summary = {
        "product_id": config.product_id,
        "walk_forward": asdict(config),
        "strategy_config": _strategy_config_to_dict(strategy_config),
        "base_variant": _variant_to_dict(base_variant),
        "regime_filter_definition": {
            "features": ["direction", "volume_ratio", "breakout_distance_norm", "box_range_atr_ratio"],
            "volume_buckets": list(REGIME_BUCKET_LABELS),
            "breakout_buckets": list(REGIME_BUCKET_LABELS),
            "box_range_buckets": list(REGIME_BUCKET_LABELS),
            "regime_min_samples": config.regime_min_samples,
            "regime_min_profit_factor": config.regime_min_profit_factor,
        },
        "baseline_summary": _variant_summary(
            trades=baseline_trades,
            equity=baseline_equity,
            starting_cash=config.starting_cash,
            strategy_config=strategy_config,
            variant=base_variant,
            entry_liquidity="taker",
            exit_liquidity="taker",
            status="ok",
            note="",
        ),
        "regime_filtered_summary": _variant_summary(
            trades=filtered_trades,
            equity=filtered_equity,
            starting_cash=config.starting_cash,
            strategy_config=strategy_config,
            variant=filtered_variant,
            entry_liquidity="taker",
            exit_liquidity="taker",
            status="ok",
            note="Adaptive regime filter learned from train-fold breakout history",
        ),
    }
    return RegimeFilterRun(
        summary=summary,
        fold_metrics=pd.DataFrame(fold_rows),
        baseline_trades=baseline_trades,
        baseline_equity=baseline_equity,
        baseline_monthly_returns=baseline_monthly_returns,
        regime_filtered_trades=filtered_trades,
        regime_filtered_equity=filtered_equity,
        regime_filtered_monthly_returns=filtered_monthly_returns,
        train_regime_table=pd.DataFrame(regime_rows),
    )


def evaluate_variant_walk_forward(
    *,
    candles: pd.DataFrame,
    optimization_config: OptimizationConfig,
    strategy_config: BreakoutConfig,
    variant: ResearchVariant,
    step_name: str,
    windows: list[WalkForwardWindow],
    entry_liquidity: str = "taker",
    exit_liquidity: str = "taker",
) -> VariantEvaluation:
    strategy_config = normalize_config(strategy_config)
    test_cash = optimization_config.starting_cash
    fold_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    selected_thresholds: list[float] = []
    status = "ok"
    note = ""

    for window in windows:
        base_variant = replace(variant, ml_gate_enabled=False, ml_gate_threshold=0.0)
        shared_backtest = _build_backtest_config(
            starting_cash=optimization_config.starting_cash,
            leverage=optimization_config.leverage,
            maker_fee_rate=optimization_config.maker_fee_rate,
            taker_fee_rate=optimization_config.taker_fee_rate,
            slippage_bps=optimization_config.slippage_bps,
            entry_liquidity=entry_liquidity,
            exit_liquidity=exit_liquidity,
        )
        train_backtest = _run_window_backtest(
            product_id=optimization_config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest,
            window_start=window.train_start,
            window_end=window.train_end,
            research_variant=base_variant,
        )
        validation_backtest = _run_window_backtest(
            product_id=optimization_config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest,
            window_start=window.validation_start,
            window_end=window.validation_end,
            research_variant=base_variant,
        )

        effective_variant = variant
        validation_result = validation_backtest.result
        test_signal_probabilities = None
        selected_threshold: Optional[float] = None

        if variant.ml_gate_enabled:
            try:
                train_labels = build_trade_label_frame(
                    trades_frame=train_backtest.result.trades_frame,
                    candidate_features=build_candidate_feature_frame(train_backtest.feature_frame),
                )
                model = fit_xgboost_model(train_labels)
                validation_predictions = predict_candidate_probabilities(
                    model,
                    build_candidate_feature_frame(validation_backtest.feature_frame),
                )
                selected_threshold, validation_result = _select_ml_threshold(
                    variant=variant,
                    prediction_map=validation_predictions,
                    product_id=optimization_config.product_id,
                    candles=candles,
                    strategy_config=strategy_config,
                    backtest_config=shared_backtest,
                    window_start=window.validation_start,
                    window_end=window.validation_end,
                )
                effective_variant = replace(variant, ml_gate_threshold=selected_threshold)
                selected_thresholds.append(selected_threshold)
                test_signal_probabilities = predict_candidate_probabilities(
                    model,
                    build_candidate_feature_frame(
                        _window_backtest_feature_frame(
                            candles=candles,
                            strategy_config=strategy_config,
                            window_start=window.test_start,
                            window_end=window.test_end,
                        )
                    ),
                )
            except ImportError as exc:
                status = "skipped_missing_ml_dependency"
                note = str(exc)
                effective_variant = base_variant

        test_backtest = _run_window_backtest(
            product_id=optimization_config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=_build_backtest_config(
                starting_cash=test_cash,
                leverage=optimization_config.leverage,
                maker_fee_rate=optimization_config.maker_fee_rate,
                taker_fee_rate=optimization_config.taker_fee_rate,
                slippage_bps=optimization_config.slippage_bps,
                entry_liquidity=entry_liquidity,
                exit_liquidity=exit_liquidity,
            ),
            window_start=window.test_start,
            window_end=window.test_end,
            research_variant=effective_variant,
            signal_probabilities=test_signal_probabilities,
        )
        test_cash = test_backtest.result.final_equity

        trade_frames.append(_attach_fold_column(test_backtest.result.trades_frame, window.fold_index))
        equity_frames.append(_attach_fold_column(test_backtest.result.equity_curve.reset_index(), window.fold_index))
        fold_rows.append(
            {
                "fold_index": window.fold_index,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "context_granularity": strategy_config.context_granularity,
                "side_mode": variant.side_mode,
                "stop_family": variant.stop_family,
                "stop_buffer_atr": round(float(variant.stop_buffer_atr), 8),
                "selected_ml_threshold": selected_threshold,
                "validation_return_pct": round(validation_result.total_return_pct, 8),
                "validation_profit_factor": round(validation_result.profit_factor, 8),
                "validation_trades_count": validation_result.trades_count,
                "validation_max_drawdown_pct": round(validation_result.max_drawdown_pct, 8),
                "test_return_pct": round(test_backtest.result.total_return_pct, 8),
                "test_profit_factor": round(test_backtest.result.profit_factor, 8),
                "test_trades_count": test_backtest.result.trades_count,
                "test_final_equity": round(test_backtest.result.final_equity, 8),
            }
        )

    trades = _concat_frames(trade_frames)
    equity = _concat_frames(equity_frames)
    monthly_returns = monthly_returns_from_equity(equity)
    summary = _variant_summary(
        trades=trades,
        equity=equity,
        starting_cash=optimization_config.starting_cash,
        strategy_config=strategy_config,
        variant=variant,
        entry_liquidity=entry_liquidity,
        exit_liquidity=exit_liquidity,
        status=status,
        note=note,
    )
    return VariantEvaluation(
        step_name=step_name,
        variant=variant,
        strategy_config=strategy_config,
        summary=summary,
        fold_metrics=pd.DataFrame(fold_rows),
        trades=trades,
        equity=equity,
        monthly_returns=monthly_returns,
        entry_liquidity=entry_liquidity,
        exit_liquidity=exit_liquidity,
        selected_thresholds=selected_thresholds,
        status=status,
        note=note,
    )


def _run_window_backtest(
    *,
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    backtest_config: BacktestConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    research_variant: ResearchVariant,
    signal_probabilities: Optional[dict[str, float]] = None,
    entry_gate: Optional[Callable[[pd.Series, pd.Timestamp], bool]] = None,
) -> _WindowBacktest:
    window_candles = _window_candles(
        candles=candles,
        strategy_config=strategy_config,
        window_start=window_start,
        window_end=window_end,
    )
    feature_frame = build_signal_frame(window_candles, strategy_config)
    result = run_backtest_for_asset(
        product_id=product_id,
        candles=window_candles,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        granularity=strategy_config.signal_granularity,
        feature_frame=feature_frame,
        entry_gate=entry_gate,
        entry_start_time=window_start,
        research_variant=research_variant,
        signal_probabilities=signal_probabilities,
    )
    return _WindowBacktest(result=result, feature_frame=feature_frame)


def _window_backtest_feature_frame(
    *,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    return build_signal_frame(
        _window_candles(
            candles=candles,
            strategy_config=strategy_config,
            window_start=window_start,
            window_end=window_end,
        ),
        strategy_config,
    )


def _window_candles(
    *,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    normalized = normalize_config(strategy_config)
    signal_seconds = GRANULARITY_TO_SECONDS[normalized.signal_granularity]
    warmup_seconds = required_history_bars(normalized) * signal_seconds
    history_start = max(
        _coerce_utc_timestamp(candles.index.min()),
        _coerce_utc_timestamp(window_start) - pd.to_timedelta(warmup_seconds, unit="s"),
    )
    return candles[(candles.index >= history_start) & (candles.index < window_end)].copy()


def _build_backtest_config(
    *,
    starting_cash: float,
    leverage: float,
    maker_fee_rate: float,
    taker_fee_rate: float,
    slippage_bps: float,
    entry_liquidity: str,
    exit_liquidity: str,
) -> BacktestConfig:
    return BacktestConfig(
        starting_cash=starting_cash,
        leverage=leverage,
        maker_fee_rate=maker_fee_rate,
        taker_fee_rate=taker_fee_rate,
        entry_liquidity=entry_liquidity,
        exit_liquidity=exit_liquidity,
        slippage_bps=slippage_bps,
    )


def _select_ml_threshold(
    *,
    variant: ResearchVariant,
    prediction_map: dict[str, float],
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    backtest_config: BacktestConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> tuple[float, AssetBacktestResult]:
    thresholds = [variant.ml_gate_threshold] if variant.ml_gate_threshold > 0 else list(THRESHOLD_GRID)
    best_threshold = thresholds[0]
    best_result: Optional[AssetBacktestResult] = None
    best_key: Optional[tuple[float, float, int]] = None
    for threshold in thresholds:
        candidate_variant = replace(variant, ml_gate_threshold=float(threshold))
        candidate = _run_window_backtest(
            product_id=product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            window_start=window_start,
            window_end=window_end,
            research_variant=candidate_variant,
            signal_probabilities=prediction_map,
        ).result
        key = (
            float(candidate.total_return_pct),
            float(candidate.profit_factor),
            -int(candidate.trades_count),
        )
        if best_result is None or key > best_key:
            best_threshold = float(threshold)
            best_result = candidate
            best_key = key
    assert best_result is not None
    return best_threshold, best_result


def _fit_regime_filter_model(
    *,
    candidate_features: pd.DataFrame,
    outcome_frame: pd.DataFrame,
    min_samples: int,
    min_profit_factor: float,
) -> tuple[RegimeFilterModel, pd.DataFrame]:
    model = RegimeFilterModel(
        volume_edges=_quantile_edges(candidate_features.get("volume_ratio")),
        breakout_edges=_quantile_edges(candidate_features.get("breakout_distance_norm")),
        box_range_edges=_quantile_edges(candidate_features.get("box_range_atr_ratio")),
        allowed_regimes=frozenset(),
    )
    if outcome_frame.empty:
        return model, pd.DataFrame(
            columns=[
                "direction",
                "volume_bucket",
                "breakout_bucket",
                "box_range_bucket",
                "trades_count",
                "win_rate_pct",
                "avg_pnl",
                "profit_factor",
                "allowed",
            ]
        )

    bucketed = _bucket_regime_frame(outcome_frame, model=model)
    grouped = (
        bucketed.groupby(
            ["direction", "volume_bucket", "breakout_bucket", "box_range_bucket"],
            dropna=False,
        )
        .agg(
            trades_count=("pnl", "size"),
            wins=("label", "sum"),
            avg_pnl=("pnl", "mean"),
            gross_profit=("pnl", lambda values: float(values[values > 0].sum()) if len(values) else 0.0),
            gross_loss=("pnl", lambda values: abs(float(values[values < 0].sum())) if len(values) else 0.0),
        )
        .reset_index()
    )
    grouped["win_rate_pct"] = grouped.apply(
        lambda row: (float(row["wins"]) / float(row["trades_count"])) * 100.0 if float(row["trades_count"]) else 0.0,
        axis=1,
    )
    grouped["profit_factor"] = grouped.apply(
        lambda row: float(row["gross_profit"]) if float(row["gross_loss"]) == 0.0 and float(row["gross_profit"]) > 0.0 else (
            float(row["gross_profit"]) / float(row["gross_loss"]) if float(row["gross_loss"]) > 0.0 else 0.0
        ),
        axis=1,
    )
    grouped["allowed"] = (
        (grouped["trades_count"] >= int(min_samples))
        & (grouped["avg_pnl"] > 0.0)
        & (grouped["profit_factor"] > float(min_profit_factor))
    )
    allowed_regimes = frozenset(
        (
            str(row.direction),
            str(row.volume_bucket),
            str(row.breakout_bucket),
            str(row.box_range_bucket),
        )
        for row in grouped.itertuples()
        if bool(row.allowed)
    )
    model = RegimeFilterModel(
        volume_edges=model.volume_edges,
        breakout_edges=model.breakout_edges,
        box_range_edges=model.box_range_edges,
        allowed_regimes=allowed_regimes,
    )
    grouped = grouped.drop(columns=["wins", "gross_profit", "gross_loss"])
    grouped["avg_pnl"] = grouped["avg_pnl"].round(8)
    grouped["win_rate_pct"] = grouped["win_rate_pct"].round(8)
    grouped["profit_factor"] = grouped["profit_factor"].round(8)
    return model, grouped


def _allowed_signal_times_for_regime_filter(
    *,
    candidate_features: pd.DataFrame,
    model: RegimeFilterModel,
) -> set[str]:
    if candidate_features.empty or not model.allowed_regimes:
        return set()
    bucketed = _bucket_regime_frame(candidate_features, model=model)
    allowed = bucketed[
        bucketed.apply(
            lambda row: (
                str(row["direction"]),
                str(row["volume_bucket"]),
                str(row["breakout_bucket"]),
                str(row["box_range_bucket"]),
            )
            in model.allowed_regimes,
            axis=1,
        )
    ]
    return set(allowed["signal_time"].astype(str).tolist())


def _bucket_regime_frame(
    frame: pd.DataFrame,
    *,
    model: RegimeFilterModel,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    bucketed = frame.copy()
    bucketed["direction"] = bucketed["direction"].astype(str)
    bucketed["volume_bucket"] = bucketed["volume_ratio"].map(
        lambda value: _bucket_label(value, model.volume_edges)
    )
    bucketed["breakout_bucket"] = bucketed["breakout_distance_norm"].map(
        lambda value: _bucket_label(value, model.breakout_edges)
    )
    bucketed["box_range_bucket"] = bucketed["box_range_atr_ratio"].map(
        lambda value: _bucket_label(value, model.box_range_edges)
    )
    return bucketed


def _bucket_label(value: Any, edges: tuple[float, float]) -> str:
    numeric = float(value)
    low_edge, high_edge = edges
    if numeric <= low_edge:
        return REGIME_BUCKET_LABELS[0]
    if numeric <= high_edge:
        return REGIME_BUCKET_LABELS[1]
    return REGIME_BUCKET_LABELS[2]


def _quantile_edges(series: Optional[pd.Series]) -> tuple[float, float]:
    if series is None:
        return (0.0, 0.0)
    clean = pd.Series(series).dropna().astype(float)
    if clean.empty:
        return (0.0, 0.0)
    low_edge = float(clean.quantile(1.0 / 3.0))
    high_edge = float(clean.quantile(2.0 / 3.0))
    if high_edge < low_edge:
        return high_edge, low_edge
    return low_edge, high_edge


def _entry_gate_from_signal_times(signal_times: set[str]) -> Callable[[pd.Series, pd.Timestamp], bool]:
    def gate(_row: pd.Series, timestamp: pd.Timestamp) -> bool:
        return timestamp.isoformat() in signal_times

    return gate


def _run_step(
    *,
    step_name: str,
    reference: VariantEvaluation,
    candidates: list[EvaluationCandidate],
    candles: pd.DataFrame,
    config: OptimizationConfig,
    windows: list[WalkForwardWindow],
) -> tuple[VariantEvaluation, tuple[list[dict[str, Any]], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    candidate_evaluations: list[VariantEvaluation] = []
    for candidate in _dedupe_candidates(candidates):
        evaluation = evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=config,
            strategy_config=candidate.strategy_config,
            variant=candidate.variant,
            step_name=step_name,
            windows=windows,
        )
        candidate_evaluations.append(evaluation)
        rows.append(
            _comparison_row(
                reference=reference,
                candidate=evaluation,
                promoted=False,
                acceptance=_acceptance_assessment(evaluation, config),
            )
        )

    promoted_candidates = [
        evaluation
        for evaluation in candidate_evaluations
        if _candidate_can_promote(reference, evaluation, config)
    ]
    if promoted_candidates:
        selected = sorted(
            promoted_candidates,
            key=lambda evaluation: (
                float(evaluation.summary["total_return_pct"]),
                float(evaluation.summary["profit_factor"]),
                -float(evaluation.summary["max_drawdown_pct"]),
                -int(evaluation.summary["trades_count"]),
                -_variant_complexity(evaluation.variant),
            ),
            reverse=True,
        )[0]
        reason = "Promoted because out-of-sample return improved and the candidate passed all acceptance checks"
    else:
        selected = reference
        reason = "No candidate exceeded the current best while staying inside the risk and quality guards"

    for row in rows:
        row["promoted"] = selected is not reference and (
            row["variant_name"] == selected.variant.name
            and row["context_granularity"] == selected.strategy_config.context_granularity
        )
    decision = {
        "step_name": step_name,
        "selected_variant": selected.variant.name,
        "selected_context_granularity": selected.strategy_config.context_granularity,
        "promoted": selected is not reference,
        "reason": reason,
    }
    return selected, (rows, decision)


def _candidate_can_promote(
    reference: VariantEvaluation,
    candidate: VariantEvaluation,
    config: OptimizationConfig,
) -> bool:
    accepted, _reason = _acceptance_assessment(candidate, config)
    if not accepted:
        return False
    return float(candidate.summary["total_return_pct"]) > float(reference.summary["total_return_pct"])


def _acceptance_assessment(
    candidate: VariantEvaluation,
    config: OptimizationConfig,
) -> tuple[bool, str]:
    if candidate.status != "ok":
        return False, candidate.note or "Candidate evaluation is not in ok status"
    if not candidate.fold_metrics.empty and (
        candidate.fold_metrics["validation_trades_count"] < config.min_validation_trades_per_fold
    ).any():
        return (
            False,
            f"Validation trades fell below {config.min_validation_trades_per_fold} in at least one fold",
        )
    if float(candidate.summary["max_drawdown_pct"]) > float(config.max_drawdown_limit_pct):
        return False, f"Max drawdown exceeded {config.max_drawdown_limit_pct:.2f}%"
    if float(candidate.summary["profit_factor"]) <= float(config.min_profit_factor):
        return False, f"Profit factor did not clear {config.min_profit_factor:.2f}"
    return True, "Accepted"


def _comparison_row(
    *,
    reference: VariantEvaluation,
    candidate: VariantEvaluation,
    promoted: bool,
    acceptance: tuple[bool, str],
) -> dict[str, Any]:
    accepted, acceptance_reason = acceptance
    return {
        "step_name": candidate.step_name,
        "reference_variant": reference.variant.name,
        "reference_context_granularity": reference.strategy_config.context_granularity,
        "variant_name": candidate.variant.name,
        "context_granularity": candidate.strategy_config.context_granularity,
        "status": candidate.status,
        "note": candidate.note,
        "accepted": accepted,
        "acceptance_reason": acceptance_reason,
        "promoted": promoted,
        "total_return_pct": round(float(candidate.summary["total_return_pct"]), 8),
        "profit_factor": round(float(candidate.summary["profit_factor"]), 8),
        "max_drawdown_pct": round(float(candidate.summary["max_drawdown_pct"]), 8),
        "trades_count": int(candidate.summary["trades_count"]),
        "fees_paid": round(float(candidate.summary["fees_paid"]), 8),
        "return_delta_vs_reference": round(
            float(candidate.summary["total_return_pct"]) - float(reference.summary["total_return_pct"]),
            8,
        ),
        "profit_factor_delta_vs_reference": round(
            float(candidate.summary["profit_factor"]) - float(reference.summary["profit_factor"]),
            8,
        ),
        "drawdown_delta_vs_reference": round(
            float(candidate.summary["max_drawdown_pct"]) - float(reference.summary["max_drawdown_pct"]),
            8,
        ),
        "long_trades": int(candidate.summary["long_trades"]),
        "long_win_rate_pct": round(float(candidate.summary["long_win_rate_pct"]), 8),
        "long_net_pnl": round(float(candidate.summary["long_net_pnl"]), 8),
        "short_trades": int(candidate.summary["short_trades"]),
        "short_win_rate_pct": round(float(candidate.summary["short_win_rate_pct"]), 8),
        "short_net_pnl": round(float(candidate.summary["short_net_pnl"]), 8),
        "side_mode": candidate.variant.side_mode,
        "blocked_utc_hours": ",".join(str(hour) for hour in candidate.variant.blocked_utc_hours),
        "cooldown_minutes": int(candidate.variant.cooldown_minutes),
        "one_trade_per_box": bool(candidate.variant.one_trade_per_box),
        "min_breakout_distance_box_ratio": float(candidate.variant.min_breakout_distance_box_ratio),
        "kalman_slope_threshold": float(candidate.variant.kalman_slope_threshold),
        "long_kalman_slope_threshold": candidate.variant.long_kalman_slope_threshold,
        "short_kalman_slope_threshold": candidate.variant.short_kalman_slope_threshold,
        "stop_family": candidate.variant.stop_family,
        "stop_buffer_atr": round(float(candidate.variant.stop_buffer_atr), 8),
        "long_reward_risk_ratio": candidate.variant.long_reward_risk_ratio,
        "short_reward_risk_ratio": candidate.variant.short_reward_risk_ratio,
        "long_min_box_atr_ratio": candidate.variant.long_min_box_atr_ratio,
        "short_min_box_atr_ratio": candidate.variant.short_min_box_atr_ratio,
        "long_min_volume_ratio": candidate.variant.long_min_volume_ratio,
        "short_min_volume_ratio": candidate.variant.short_min_volume_ratio,
        "long_min_breakout_distance_box_ratio": candidate.variant.long_min_breakout_distance_box_ratio,
        "short_min_breakout_distance_box_ratio": candidate.variant.short_min_breakout_distance_box_ratio,
        "time_stop_bars": candidate.variant.time_stop_bars,
        "breakeven_trigger_r": candidate.variant.breakeven_trigger_r,
        "entry_liquidity": candidate.entry_liquidity,
        "exit_liquidity": candidate.exit_liquidity,
    }


def _side_selection_candidates(
    reference: VariantEvaluation,
    config: OptimizationConfig,
) -> list[EvaluationCandidate]:
    candidates = []
    for side_mode in config.side_modes:
        if side_mode == reference.variant.side_mode:
            continue
        candidates.append(
            EvaluationCandidate(
                strategy_config=reference.strategy_config,
                variant=replace(reference.variant, name=f"side_{side_mode}", side_mode=side_mode),
            )
        )
    return candidates


def _context_timeframe_candidates(
    reference: VariantEvaluation,
    config: OptimizationConfig,
) -> list[EvaluationCandidate]:
    candidates = []
    for context_granularity in config.context_granularities:
        if context_granularity == reference.strategy_config.context_granularity:
            continue
        candidates.append(
            EvaluationCandidate(
                strategy_config=replace(
                    reference.strategy_config,
                    signal_granularity="ONE_MINUTE",
                    context_granularity=context_granularity,
                ),
                variant=replace(
                    reference.variant,
                    name=f"{reference.variant.name}_{_granularity_label(context_granularity)}",
                ),
            )
        )
    return candidates


def _trade_suppression_candidates(
    reference: VariantEvaluation,
    config: OptimizationConfig,
) -> list[EvaluationCandidate]:
    bottom_hours = _bottom_hours_by_avg_pnl(reference.trades)
    candidates: list[EvaluationCandidate] = []

    for count in config.blocked_hour_counts:
        if count <= 0 or len(bottom_hours) < count:
            continue
        candidates.append(
            EvaluationCandidate(
                strategy_config=reference.strategy_config,
                variant=replace(
                    reference.variant,
                    name=f"block_bottom_{count}_hours",
                    blocked_utc_hours=tuple(bottom_hours[:count]),
                ),
            )
        )

    for cooldown, one_trade_per_box in product(
        config.cooldown_grid,
        config.one_trade_per_box_options,
    ):
        if (
            int(cooldown) == int(reference.variant.cooldown_minutes)
            and bool(one_trade_per_box) == bool(reference.variant.one_trade_per_box)
        ):
            continue
        name = f"cooldown_{cooldown}m"
        if one_trade_per_box:
            name += "_one_box"
        candidates.append(
            EvaluationCandidate(
                strategy_config=reference.strategy_config,
                variant=replace(
                    reference.variant,
                    name=name,
                    cooldown_minutes=int(cooldown),
                    one_trade_per_box=bool(one_trade_per_box),
                ),
            )
        )

    combo_cooldown = 5 if 5 in config.cooldown_grid else (config.cooldown_grid[0] if config.cooldown_grid else 0)
    for count in config.blocked_hour_counts:
        if count <= 0 or len(bottom_hours) < count:
            continue
        candidates.append(
            EvaluationCandidate(
                strategy_config=reference.strategy_config,
                variant=replace(
                    reference.variant,
                    name=f"block_{count}_hours_cooldown_one_box",
                    blocked_utc_hours=tuple(bottom_hours[:count]),
                    cooldown_minutes=int(combo_cooldown),
                    one_trade_per_box=True,
                ),
            )
        )

    return candidates


def _entry_quality_candidates(
    reference: VariantEvaluation,
    config: OptimizationConfig,
) -> list[EvaluationCandidate]:
    candidates: list[EvaluationCandidate] = []
    variant = reference.variant
    strategy_config = reference.strategy_config
    moderate_box = _grid_pick(config.min_box_atr_grid, 1)
    strong_box = _grid_pick(config.min_box_atr_grid, len(config.min_box_atr_grid) - 1)
    moderate_volume = _grid_pick(config.min_volume_ratio_grid, 1)
    strong_volume = _grid_pick(config.min_volume_ratio_grid, len(config.min_volume_ratio_grid) - 1)
    moderate_breakout = _grid_pick(config.breakout_distance_grid, 1)
    strong_breakout = _grid_pick(config.breakout_distance_grid, len(config.breakout_distance_grid) - 1)
    moderate_slope = _grid_pick(config.kalman_slope_threshold_grid, 1)
    strong_slope = _grid_pick(config.kalman_slope_threshold_grid, len(config.kalman_slope_threshold_grid) - 1)

    if variant.side_mode == SIDE_MODE_LONG_ONLY:
        return _single_side_entry_quality_candidates(
            side="long",
            reference_variant=variant,
            strategy_config=strategy_config,
            config=config,
            box_combo=(moderate_box, strong_box),
            volume_combo=(moderate_volume, strong_volume),
            breakout_combo=(moderate_breakout, strong_breakout),
            slope_combo=(moderate_slope, strong_slope),
        )

    if variant.side_mode == SIDE_MODE_SHORT_ONLY:
        return _single_side_entry_quality_candidates(
            side="short",
            reference_variant=variant,
            strategy_config=strategy_config,
            config=config,
            box_combo=(moderate_box, strong_box),
            volume_combo=(moderate_volume, strong_volume),
            breakout_combo=(moderate_breakout, strong_breakout),
            slope_combo=(moderate_slope, strong_slope),
        )

    symmetric_box = _non_current_values(config.min_box_atr_grid, strategy_config.min_box_atr_ratio)
    symmetric_volume = _non_current_values(config.min_volume_ratio_grid, strategy_config.min_volume_ratio)
    symmetric_breakout = _non_current_values(
        config.breakout_distance_grid,
        variant.min_breakout_distance_box_ratio,
    )
    symmetric_slope = _non_current_values(
        config.kalman_slope_threshold_grid,
        variant.kalman_slope_threshold,
    )

    for value in symmetric_box:
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"symmetric_box_atr_{_value_label(value)}",
                    long_min_box_atr_ratio=float(value),
                    short_min_box_atr_ratio=float(value),
                ),
            )
        )
    for value in symmetric_volume:
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"symmetric_volume_{_value_label(value)}",
                    long_min_volume_ratio=float(value),
                    short_min_volume_ratio=float(value),
                ),
            )
        )
    for value in symmetric_breakout:
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"symmetric_breakout_{_value_label(value)}",
                    long_min_breakout_distance_box_ratio=float(value),
                    short_min_breakout_distance_box_ratio=float(value),
                ),
            )
        )
    for value in symmetric_slope:
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"symmetric_kalman_slope_{_value_label(value)}",
                    kalman_slope_threshold=float(value),
                ),
            )
        )

    candidates.extend(
        [
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name="symmetric_entry_combo_moderate",
                    long_min_box_atr_ratio=moderate_box,
                    short_min_box_atr_ratio=moderate_box,
                    long_min_volume_ratio=moderate_volume,
                    short_min_volume_ratio=moderate_volume,
                    long_min_breakout_distance_box_ratio=moderate_breakout,
                    short_min_breakout_distance_box_ratio=moderate_breakout,
                    kalman_slope_threshold=moderate_slope,
                ),
            ),
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name="symmetric_entry_combo_strong",
                    long_min_box_atr_ratio=strong_box,
                    short_min_box_atr_ratio=strong_box,
                    long_min_volume_ratio=strong_volume,
                    short_min_volume_ratio=strong_volume,
                    long_min_breakout_distance_box_ratio=strong_breakout,
                    short_min_breakout_distance_box_ratio=strong_breakout,
                    kalman_slope_threshold=strong_slope,
                ),
            ),
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name="long_bias_entry_combo",
                    long_min_box_atr_ratio=strong_box,
                    long_min_volume_ratio=strong_volume,
                    long_min_breakout_distance_box_ratio=moderate_breakout,
                    long_kalman_slope_threshold=moderate_slope,
                ),
            ),
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name="short_bias_entry_combo",
                    short_min_box_atr_ratio=strong_box,
                    short_min_volume_ratio=strong_volume,
                    short_min_breakout_distance_box_ratio=moderate_breakout,
                    short_kalman_slope_threshold=moderate_slope,
                ),
            ),
        ]
    )
    return candidates


def _single_side_entry_quality_candidates(
    *,
    side: str,
    reference_variant: ResearchVariant,
    strategy_config: BreakoutConfig,
    config: OptimizationConfig,
    box_combo: tuple[float, float],
    volume_combo: tuple[float, float],
    breakout_combo: tuple[float, float],
    slope_combo: tuple[float, float],
) -> list[EvaluationCandidate]:
    candidates: list[EvaluationCandidate] = []
    if side == "long":
        current_box = _effective_directional_value(
            strategy_config.min_box_atr_ratio,
            reference_variant.long_min_box_atr_ratio,
        )
        current_volume = _effective_directional_value(
            strategy_config.min_volume_ratio,
            reference_variant.long_min_volume_ratio,
        )
        current_breakout = _effective_directional_value(
            reference_variant.min_breakout_distance_box_ratio,
            reference_variant.long_min_breakout_distance_box_ratio,
        )
        current_slope = _effective_directional_value(
            reference_variant.kalman_slope_threshold,
            reference_variant.long_kalman_slope_threshold,
        )
        for value in _non_current_values(config.min_box_atr_grid, current_box):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name=f"long_box_atr_{_value_label(value)}",
                        long_min_box_atr_ratio=float(value),
                    ),
                )
            )
        for value in _non_current_values(config.min_volume_ratio_grid, current_volume):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name=f"long_volume_{_value_label(value)}",
                        long_min_volume_ratio=float(value),
                    ),
                )
            )
        for value in _non_current_values(config.breakout_distance_grid, current_breakout):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name=f"long_breakout_{_value_label(value)}",
                        long_min_breakout_distance_box_ratio=float(value),
                    ),
                )
            )
        for value in _non_current_values(config.kalman_slope_threshold_grid, current_slope):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name=f"long_kalman_slope_{_value_label(value)}",
                        long_kalman_slope_threshold=float(value),
                    ),
                )
            )
        candidates.extend(
            [
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name="long_entry_combo_moderate",
                        long_min_box_atr_ratio=box_combo[0],
                        long_min_volume_ratio=volume_combo[0],
                        long_min_breakout_distance_box_ratio=breakout_combo[0],
                        long_kalman_slope_threshold=slope_combo[0],
                    ),
                ),
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        reference_variant,
                        name="long_entry_combo_strong",
                        long_min_box_atr_ratio=box_combo[1],
                        long_min_volume_ratio=volume_combo[1],
                        long_min_breakout_distance_box_ratio=breakout_combo[1],
                        long_kalman_slope_threshold=slope_combo[1],
                    ),
                ),
            ]
        )
        return candidates

    current_box = _effective_directional_value(
        strategy_config.min_box_atr_ratio,
        reference_variant.short_min_box_atr_ratio,
    )
    current_volume = _effective_directional_value(
        strategy_config.min_volume_ratio,
        reference_variant.short_min_volume_ratio,
    )
    current_breakout = _effective_directional_value(
        reference_variant.min_breakout_distance_box_ratio,
        reference_variant.short_min_breakout_distance_box_ratio,
    )
    current_slope = _effective_directional_value(
        reference_variant.kalman_slope_threshold,
        reference_variant.short_kalman_slope_threshold,
    )
    for value in _non_current_values(config.min_box_atr_grid, current_box):
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name=f"short_box_atr_{_value_label(value)}",
                    short_min_box_atr_ratio=float(value),
                ),
            )
        )
    for value in _non_current_values(config.min_volume_ratio_grid, current_volume):
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name=f"short_volume_{_value_label(value)}",
                    short_min_volume_ratio=float(value),
                ),
            )
        )
    for value in _non_current_values(config.breakout_distance_grid, current_breakout):
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name=f"short_breakout_{_value_label(value)}",
                    short_min_breakout_distance_box_ratio=float(value),
                ),
            )
        )
    for value in _non_current_values(config.kalman_slope_threshold_grid, current_slope):
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name=f"short_kalman_slope_{_value_label(value)}",
                    short_kalman_slope_threshold=float(value),
                ),
            )
        )
    candidates.extend(
        [
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name="short_entry_combo_moderate",
                    short_min_box_atr_ratio=box_combo[0],
                    short_min_volume_ratio=volume_combo[0],
                    short_min_breakout_distance_box_ratio=breakout_combo[0],
                    short_kalman_slope_threshold=slope_combo[0],
                ),
            ),
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    reference_variant,
                    name="short_entry_combo_strong",
                    short_min_box_atr_ratio=box_combo[1],
                    short_min_volume_ratio=volume_combo[1],
                    short_min_breakout_distance_box_ratio=breakout_combo[1],
                    short_kalman_slope_threshold=slope_combo[1],
                ),
            ),
        ]
    )
    return candidates


def _exit_redesign_candidates(
    reference: VariantEvaluation,
    config: OptimizationConfig,
) -> list[EvaluationCandidate]:
    candidates: list[EvaluationCandidate] = []
    variant = reference.variant
    strategy_config = reference.strategy_config

    for stop_family in config.stop_families:
        if stop_family == STOP_FAMILY_STRUCTURAL_ATR_BUFFER:
            for stop_buffer_atr in config.stop_buffer_atr_grid:
                if (
                    stop_family == variant.stop_family
                    and float(stop_buffer_atr) == float(variant.stop_buffer_atr)
                ):
                    continue
                candidates.append(
                    EvaluationCandidate(
                        strategy_config=strategy_config,
                        variant=replace(
                            variant,
                            name=f"structural_atr_buffer_{_value_label(stop_buffer_atr)}",
                            stop_family=stop_family,
                            stop_buffer_atr=float(stop_buffer_atr),
                        ),
                    )
                )
            continue
        if stop_family == variant.stop_family and float(variant.stop_buffer_atr) == 0.0:
            continue
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=stop_family,
                    stop_family=stop_family,
                    stop_buffer_atr=0.0,
                ),
            )
        )

    if variant.side_mode == SIDE_MODE_LONG_ONLY:
        current_rr = _effective_directional_value(
            strategy_config.reward_risk_ratio,
            variant.long_reward_risk_ratio,
        )
        for reward_risk_ratio in _non_current_values(config.long_reward_risk_grid, current_rr):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        variant,
                        name=f"long_rr_{_value_label(reward_risk_ratio)}",
                        long_reward_risk_ratio=float(reward_risk_ratio),
                    ),
                )
            )
    elif variant.side_mode == SIDE_MODE_SHORT_ONLY:
        current_rr = _effective_directional_value(
            strategy_config.reward_risk_ratio,
            variant.short_reward_risk_ratio,
        )
        for reward_risk_ratio in _non_current_values(config.short_reward_risk_grid, current_rr):
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        variant,
                        name=f"short_rr_{_value_label(reward_risk_ratio)}",
                        short_reward_risk_ratio=float(reward_risk_ratio),
                    ),
                )
            )
    else:
        current_long_rr = _effective_directional_value(
            strategy_config.reward_risk_ratio,
            variant.long_reward_risk_ratio,
        )
        current_short_rr = _effective_directional_value(
            strategy_config.reward_risk_ratio,
            variant.short_reward_risk_ratio,
        )
        for long_rr, short_rr in product(config.long_reward_risk_grid, config.short_reward_risk_grid):
            if float(long_rr) == float(current_long_rr) and float(short_rr) == float(current_short_rr):
                continue
            candidates.append(
                EvaluationCandidate(
                    strategy_config=strategy_config,
                    variant=replace(
                        variant,
                        name=f"rr_long_{_value_label(long_rr)}_short_{_value_label(short_rr)}",
                        long_reward_risk_ratio=float(long_rr),
                        short_reward_risk_ratio=float(short_rr),
                    ),
                )
            )

    for time_stop_bars in config.time_stop_grid:
        if time_stop_bars == variant.time_stop_bars:
            continue
        label = "none" if time_stop_bars is None else str(time_stop_bars)
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"time_stop_{label}",
                    time_stop_bars=time_stop_bars,
                ),
            )
        )

    for breakeven_trigger_r in config.breakeven_grid:
        if breakeven_trigger_r == variant.breakeven_trigger_r:
            continue
        label = "none" if breakeven_trigger_r is None else _value_label(breakeven_trigger_r)
        candidates.append(
            EvaluationCandidate(
                strategy_config=strategy_config,
                variant=replace(
                    variant,
                    name=f"breakeven_{label}",
                    breakeven_trigger_r=breakeven_trigger_r,
                ),
            )
        )

    return candidates


def _execution_sensitivity_evaluations(
    *,
    current_best: VariantEvaluation,
    candles: pd.DataFrame,
    config: OptimizationConfig,
    windows: list[WalkForwardWindow],
) -> list[VariantEvaluation]:
    evaluations = [
        evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=config,
            strategy_config=current_best.strategy_config,
            variant=replace(current_best.variant, name=f"{current_best.variant.name}_taker_taker"),
            step_name="execution_sensitivity",
            windows=windows,
            entry_liquidity="taker",
            exit_liquidity="taker",
        ),
        evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=config,
            strategy_config=current_best.strategy_config,
            variant=replace(current_best.variant, name=f"{current_best.variant.name}_maker_taker"),
            step_name="execution_sensitivity",
            windows=windows,
            entry_liquidity="maker",
            exit_liquidity="taker",
        ),
    ]

    lower_frequency_candidate = replace(
        current_best.variant,
        name=f"{current_best.variant.name}_lower_frequency_validation",
        cooldown_minutes=max(int(current_best.variant.cooldown_minutes), 5),
        one_trade_per_box=True,
    )
    evaluations.append(
        evaluate_variant_walk_forward(
            candles=candles,
            optimization_config=config,
            strategy_config=current_best.strategy_config,
            variant=lower_frequency_candidate,
            step_name="execution_sensitivity",
            windows=windows,
            entry_liquidity="taker",
            exit_liquidity="taker",
        )
    )
    return evaluations


def _variant_summary(
    *,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    starting_cash: float,
    strategy_config: BreakoutConfig,
    variant: ResearchVariant,
    entry_liquidity: str,
    exit_liquidity: str,
    status: str,
    note: str,
) -> dict[str, Any]:
    final_equity = float(equity.iloc[-1]["equity"]) if not equity.empty else starting_cash
    total_return_pct = ((final_equity / starting_cash) - 1.0) * 100.0 if starting_cash else 0.0
    max_drawdown_pct = _max_drawdown_pct(equity)
    win_rate_pct, profit_factor = _trade_stats(trades)
    fees_paid = float(trades["entry_fee"].sum() + trades["exit_fee"].sum()) if not trades.empty else 0.0
    long_stats = _side_stats(trades, "LONG")
    short_stats = _side_stats(trades, "SHORT")
    return {
        "variant_name": variant.name,
        "status": status,
        "note": note,
        "starting_cash": round(starting_cash, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": int(len(trades)),
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 4),
        "fees_paid": round(fees_paid, 4),
        "signal_granularity": strategy_config.signal_granularity,
        "context_granularity": strategy_config.context_granularity,
        "long_trades": long_stats["trades"],
        "long_win_rate_pct": round(long_stats["win_rate_pct"], 4),
        "long_net_pnl": round(long_stats["net_pnl"], 4),
        "short_trades": short_stats["trades"],
        "short_win_rate_pct": round(short_stats["win_rate_pct"], 4),
        "short_net_pnl": round(short_stats["net_pnl"], 4),
        "blocked_utc_hours": list(variant.blocked_utc_hours),
        "stop_family": variant.stop_family,
        "stop_buffer_atr": round(float(variant.stop_buffer_atr), 8),
        "entry_liquidity": entry_liquidity,
        "exit_liquidity": exit_liquidity,
    }


def _trade_stats(trades: pd.DataFrame) -> tuple[float, float]:
    if trades.empty:
        return 0.0, 0.0
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]
    win_rate_pct = (len(wins) / len(trades)) * 100.0
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    if gross_loss == 0.0:
        return win_rate_pct, gross_profit if gross_profit > 0 else 0.0
    return win_rate_pct, gross_profit / gross_loss


def _side_stats(trades: pd.DataFrame, direction: str) -> dict[str, float | int]:
    if trades.empty or "direction" not in trades.columns:
        return {"trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0}
    subset = trades[trades["direction"] == direction]
    if subset.empty:
        return {"trades": 0, "win_rate_pct": 0.0, "net_pnl": 0.0}
    wins = subset[subset["pnl"] > 0]
    return {
        "trades": int(len(subset)),
        "win_rate_pct": (len(wins) / len(subset)) * 100.0,
        "net_pnl": float(subset["pnl"].sum()),
    }


def _max_drawdown_pct(equity: pd.DataFrame) -> float:
    if equity.empty:
        return 0.0
    values = equity["equity"].astype(float)
    peaks = values.cummax()
    drawdown = values / peaks - 1.0
    return abs(float(drawdown.min()) * 100.0)


def _bottom_hours_by_avg_pnl(trades: pd.DataFrame) -> list[int]:
    if trades.empty:
        return []
    frame = trades.copy()
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
    grouped = (
        frame.groupby(frame["signal_time"].dt.hour)
        .agg(trades_count=("pnl", "size"), avg_pnl=("pnl", "mean"))
        .reset_index()
    )
    grouped = grouped.sort_values(["avg_pnl", "trades_count", "signal_time"], ascending=[True, False, True])
    return grouped["signal_time"].astype(int).tolist()


def _variant_complexity(variant: ResearchVariant) -> int:
    return sum(
        [
            int(variant.side_mode != SIDE_MODE_BOTH),
            len(variant.blocked_utc_hours),
            int(variant.cooldown_minutes > 0),
            int(variant.one_trade_per_box),
            int(variant.min_breakout_distance_box_ratio > 0),
            int(variant.kalman_slope_threshold > 0),
            int(variant.ml_gate_enabled),
            int(variant.long_reward_risk_ratio is not None),
            int(variant.short_reward_risk_ratio is not None),
            int(variant.long_min_box_atr_ratio is not None),
            int(variant.short_min_box_atr_ratio is not None),
            int(variant.long_min_volume_ratio is not None),
            int(variant.short_min_volume_ratio is not None),
            int(variant.long_min_breakout_distance_box_ratio is not None),
            int(variant.short_min_breakout_distance_box_ratio is not None),
            int(variant.long_kalman_slope_threshold is not None),
            int(variant.short_kalman_slope_threshold is not None),
            int(variant.stop_family != STOP_FAMILY_BREAKOUT_CANDLE),
            int(variant.stop_buffer_atr > 0),
            int(variant.time_stop_bars is not None),
            int(variant.breakeven_trigger_r is not None),
        ]
    )


def _variant_to_dict(variant: ResearchVariant) -> dict[str, Any]:
    payload = asdict(variant)
    payload["blocked_utc_hours"] = list(variant.blocked_utc_hours)
    return payload


def _strategy_config_to_dict(strategy_config: BreakoutConfig) -> dict[str, Any]:
    normalized = normalize_config(strategy_config)
    return {
        "timeframe_mode": normalized.timeframe_mode,
        "signal_granularity": normalized.signal_granularity,
        "context_granularity": normalized.context_granularity,
        "reward_risk_ratio": normalized.reward_risk_ratio,
        "atr_period": normalized.atr_period,
        "volume_window": normalized.volume_window,
        "min_box_atr_ratio": normalized.min_box_atr_ratio,
        "min_volume_ratio": normalized.min_volume_ratio,
        "risk_fraction": normalized.risk_fraction,
        "leverage": normalized.leverage,
        "min_position_notional": normalized.min_position_notional,
        "max_position_notional": normalized.max_position_notional,
        "kalman_process_variance": normalized.kalman_process_variance,
        "kalman_measurement_variance": normalized.kalman_measurement_variance,
    }


def _attach_fold_column(frame: pd.DataFrame, fold_index: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    attached = frame.copy()
    attached.insert(0, "fold_index", fold_index)
    return attached


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    combined = pd.concat(non_empty, ignore_index=True)
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    if "entry_time" in combined.columns:
        for column in ("signal_time", "entry_time", "exit_time"):
            combined[column] = pd.to_datetime(combined[column], utc=True).map(lambda value: value.isoformat())
        combined = combined.sort_values("entry_time").reset_index(drop=True)
    return combined


def _coerce_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _convert_windows(raw_windows: list[Any]) -> list[WalkForwardWindow]:
    return [
        WalkForwardWindow(
            fold_index=int(window.fold_index),
            train_start=_coerce_utc_timestamp(window.train_start),
            validation_start=_coerce_utc_timestamp(window.validation_start),
            test_start=_coerce_utc_timestamp(window.test_start),
            test_end=_coerce_utc_timestamp(window.test_end),
        )
        for window in raw_windows
    ]


def _default_context_granularity(config: OptimizationConfig) -> str:
    if config.context_granularities:
        return config.context_granularities[0]
    return "FIVE_MINUTE"


def _dedupe_candidates(candidates: list[EvaluationCandidate]) -> list[EvaluationCandidate]:
    deduped: list[EvaluationCandidate] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            tuple(sorted(asdict(candidate.variant).items())),
            tuple(sorted(_strategy_config_to_dict(candidate.strategy_config).items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _granularity_label(granularity: str) -> str:
    return granularity.lower().replace("_minute", "m")


def _value_label(value: float) -> str:
    return str(value).replace(".", "_")


def _grid_pick(values: tuple[float, ...], index: int) -> float:
    if not values:
        return 0.0
    bounded = max(0, min(index, len(values) - 1))
    return float(values[bounded])


def _non_current_values(values: tuple[float, ...], current_value: float) -> list[float]:
    return [float(value) for value in values if float(value) != float(current_value)]


def _effective_directional_value(base_value: float, override_value: Optional[float]) -> float:
    return float(override_value if override_value is not None else base_value)


__all__ = [
    "DEFAULT_BLOCKED_HOUR_COUNTS",
    "DEFAULT_BREAKEVEN_GRID",
    "DEFAULT_BREAKOUT_DISTANCE_GRID",
    "DEFAULT_CONTEXT_GRANULARITIES",
    "DEFAULT_COOLDOWN_GRID",
    "DEFAULT_KALMAN_SLOPE_THRESHOLD_GRID",
    "DEFAULT_LONG_REWARD_RISK_GRID",
    "DEFAULT_MIN_BOX_ATR_GRID",
    "DEFAULT_MIN_VOLUME_RATIO_GRID",
    "DEFAULT_ONE_TRADE_PER_BOX_OPTIONS",
    "DEFAULT_SHORT_REWARD_RISK_GRID",
    "DEFAULT_SIDE_MODES",
    "DEFAULT_STOP_BUFFER_ATR_GRID",
    "DEFAULT_STOP_FAMILIES",
    "DEFAULT_TIME_STOP_GRID",
    "EvaluationCandidate",
    "OptimizationConfig",
    "OptimizationRun",
    "RegimeFilterRun",
    "ResearchVariant",
    "SIDE_MODE_BOTH",
    "SIDE_MODE_LONG_ONLY",
    "SIDE_MODE_SHORT_ONLY",
    "VariantEvaluation",
    "WalkForwardWindow",
    "XGBoostFilterRun",
    "build_base_strategy_config",
    "evaluate_variant_walk_forward",
    "run_regime_filter_research",
    "run_optimization_sequence",
    "run_xgboost_filter_research",
]
