from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
import random
from typing import Any, Optional

import pandas as pd  # type: ignore

try:
    from xgboost import XGBClassifier  # type: ignore
except ImportError:  # pragma: no cover - optional dependency at import time
    XGBClassifier = None

from scalping_5min_momentum.back_testing.data_sources import load_local_coinbase_csv
from scalping_5min_momentum.back_testing.engine import (
    AssetBacktestResult,
    BacktestConfig,
    ResearchVariant,
    run_backtest_for_asset,
)
from scalping_5min_momentum.coinbase_advanced import GRANULARITY_TO_SECONDS
from scalping_5min_momentum.scalping_strategy import (
    BreakoutConfig,
    TIMEFRAME_MODE_STRICT,
    build_signal_frame,
    normalize_config,
    required_history_bars,
)

RULE_SEARCH_SPACE = {
    "reward_risk_ratio": (1.25, 1.5, 1.75, 2.0),
    "atr_period": (7, 14, 21),
    "volume_window": (10, 20, 30),
    "min_box_atr_ratio": (0.4, 0.6, 0.8, 1.0),
    "min_volume_ratio": (0.9, 1.0, 1.1, 1.2),
    "kalman_process_variance": (0.0001, 0.0005, 0.001),
    "kalman_measurement_variance": (0.005, 0.01, 0.02),
}
THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
MIN_VALIDATION_TRADES = 10
MAX_VALIDATION_DRAWDOWN_PCT = 35.0
ML_FEATURE_COLUMNS = (
    "direction_long",
    "box_range_atr_ratio",
    "volume_ratio",
    "breakout_distance_norm",
    "kalman_gap_norm",
    "kalman_slope_norm",
    "candle_range_ratio",
    "candle_body_ratio",
)
CANDIDATE_FEATURE_COLUMNS = (
    "signal_time",
    "direction",
    "direction_long",
    "box_range_atr_ratio",
    "volume_ratio",
    "breakout_distance_norm",
    "kalman_gap_norm",
    "kalman_slope_norm",
    "candle_range_ratio",
    "candle_body_ratio",
    "utc_hour",
)


@dataclass(frozen=True)
class WalkForwardConfig:
    product_id: str = "BTC-PERP"
    train_months: int = 6
    validation_months: int = 1
    test_months: int = 1
    starting_cash: float = 10000.0
    leverage: float = 2.0
    taker_fee_rate: float = 0.0006
    slippage_bps: float = 2.0
    seed: int = 42
    max_rule_configs: int = 128


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


@dataclass
class WalkForwardRun:
    summary: dict[str, Any]
    fold_metrics: pd.DataFrame
    selected_rule_configs: pd.DataFrame
    baseline_oos_trades: pd.DataFrame
    ml_gated_oos_trades: pd.DataFrame
    baseline_oos_equity: pd.DataFrame
    ml_gated_oos_equity: pd.DataFrame
    baseline_monthly_returns: pd.DataFrame
    ml_gated_monthly_returns: pd.DataFrame
    ml_feature_importance: pd.DataFrame


class ConstantProbabilityModel:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = max(0.0, min(1.0, positive_probability))
        self.feature_importances_ = [0.0] * len(ML_FEATURE_COLUMNS)

    def predict_proba(self, data: pd.DataFrame) -> list[list[float]]:
        return [[1.0 - self.positive_probability, self.positive_probability] for _ in range(len(data))]


def default_btc_csv_path(package_root: Path) -> Path:
    candidate_dir = package_root / "output_local" / "maker_maker"
    candidates = sorted(candidate_dir.glob("BTC_PERP_INTX_ONE_MINUTE_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a BTC perpetual minute CSV under {candidate_dir}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_btc_candles(csv_path: str | Path) -> pd.DataFrame:
    return load_local_coinbase_csv(csv_path)


def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    *,
    train_months: int,
    validation_months: int,
    test_months: int,
) -> list[WalkForwardWindow]:
    if index.empty:
        return []
    first_timestamp = _coerce_utc_timestamp(index.min())
    last_timestamp = _coerce_utc_timestamp(index.max())
    first_full_month = _next_month_start(first_timestamp)
    available_end = _month_start(last_timestamp)
    windows: list[WalkForwardWindow] = []
    cursor = first_full_month
    fold_index = 1
    while True:
        validation_start = cursor + pd.DateOffset(months=train_months)
        test_start = validation_start + pd.DateOffset(months=validation_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > available_end:
            break
        windows.append(
            WalkForwardWindow(
                fold_index=fold_index,
                train_start=cursor,
                validation_start=validation_start,
                test_start=test_start,
                test_end=test_end,
            )
        )
        cursor = cursor + pd.DateOffset(months=1)
        fold_index += 1
    return windows


def sample_rule_configs(
    base_config: BreakoutConfig,
    *,
    max_configs: int,
    seed: int,
) -> list[BreakoutConfig]:
    normalized = normalize_config(base_config)
    combinations: list[BreakoutConfig] = []
    for values in product(
        RULE_SEARCH_SPACE["reward_risk_ratio"],
        RULE_SEARCH_SPACE["atr_period"],
        RULE_SEARCH_SPACE["volume_window"],
        RULE_SEARCH_SPACE["min_box_atr_ratio"],
        RULE_SEARCH_SPACE["min_volume_ratio"],
        RULE_SEARCH_SPACE["kalman_process_variance"],
        RULE_SEARCH_SPACE["kalman_measurement_variance"],
    ):
        combinations.append(
            BreakoutConfig(
                timeframe_mode=normalized.timeframe_mode,
                signal_granularity=normalized.signal_granularity,
                context_granularity=normalized.context_granularity,
                reward_risk_ratio=values[0],
                atr_period=int(values[1]),
                volume_window=int(values[2]),
                min_box_atr_ratio=values[3],
                min_volume_ratio=values[4],
                risk_fraction=normalized.risk_fraction,
                leverage=normalized.leverage,
                min_position_notional=normalized.min_position_notional,
                max_position_notional=normalized.max_position_notional,
                kalman_process_variance=values[5],
                kalman_measurement_variance=values[6],
            )
        )

    unique_configs = list(dict.fromkeys(combinations))
    rng = random.Random(seed)
    default_config = normalize_config(base_config)
    remaining = [config for config in unique_configs if config != default_config]
    sample_size = max(max_configs - 1, 0)
    if sample_size >= len(remaining):
        sampled = remaining
    else:
        sampled = rng.sample(remaining, sample_size)
    return [default_config, *sampled]


def run_walk_forward_research(
    candles: pd.DataFrame,
    config: WalkForwardConfig,
    *,
    windows: Optional[list[WalkForwardWindow]] = None,
) -> WalkForwardRun:
    candles = candles.sort_index()
    base_strategy_config = normalize_config(
        BreakoutConfig(
            timeframe_mode=TIMEFRAME_MODE_STRICT,
            signal_granularity="ONE_MINUTE",
            context_granularity="FIVE_MINUTE",
            leverage=config.leverage,
        )
    )
    windows = windows or generate_walk_forward_windows(
        candles.index,
        train_months=config.train_months,
        validation_months=config.validation_months,
        test_months=config.test_months,
    )
    if not windows:
        raise ValueError("No walk-forward windows could be generated from the provided BTC data")

    sampled_configs = sample_rule_configs(
        base_strategy_config,
        max_configs=config.max_rule_configs,
        seed=config.seed,
    )

    baseline_cash = config.starting_cash
    ml_cash = config.starting_cash
    fold_metrics_rows: list[dict[str, Any]] = []
    selected_config_rows: list[dict[str, Any]] = []
    baseline_trade_frames: list[pd.DataFrame] = []
    ml_trade_frames: list[pd.DataFrame] = []
    baseline_equity_frames: list[pd.DataFrame] = []
    ml_equity_frames: list[pd.DataFrame] = []
    feature_importance_rows: list[dict[str, Any]] = []

    for window in windows:
        selected_config, selection_summary = _select_rule_config(
            candles=candles,
            window=window,
            strategy_configs=sampled_configs,
            research_config=config,
        )

        shared_backtest_config = _build_backtest_config(
            starting_cash=config.starting_cash,
            leverage=config.leverage,
            taker_fee_rate=config.taker_fee_rate,
            slippage_bps=config.slippage_bps,
        )
        train_backtest = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=selected_config,
            backtest_config=shared_backtest_config,
            window_start=window.train_start,
            window_end=window.train_end,
        )
        validation_backtest = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=selected_config,
            backtest_config=shared_backtest_config,
            window_start=window.validation_start,
            window_end=window.validation_end,
        )
        train_labels = build_trade_label_frame(
            trades_frame=train_backtest.result.trades_frame,
            candidate_features=build_candidate_feature_frame(train_backtest.feature_frame),
        )
        model = fit_xgboost_model(train_labels)
        validation_candidates = build_candidate_feature_frame(validation_backtest.feature_frame)
        validation_predictions = predict_candidate_probabilities(model, validation_candidates)
        threshold, validation_ml_result = _select_threshold(
            product_id=config.product_id,
            candles=candles,
            strategy_config=selected_config,
            backtest_config=shared_backtest_config,
            window_start=window.validation_start,
            window_end=window.validation_end,
            prediction_map=validation_predictions,
        )

        baseline_test = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=selected_config,
            backtest_config=_build_backtest_config(
                starting_cash=baseline_cash,
                leverage=config.leverage,
                taker_fee_rate=config.taker_fee_rate,
                slippage_bps=config.slippage_bps,
            ),
            window_start=window.test_start,
            window_end=window.test_end,
        )
        test_predictions = predict_candidate_probabilities(
            model,
            build_candidate_feature_frame(baseline_test.feature_frame),
        )
        ml_test = _run_window_backtest(
            product_id=config.product_id,
            candles=candles,
            strategy_config=selected_config,
            backtest_config=_build_backtest_config(
                starting_cash=ml_cash,
                leverage=config.leverage,
                taker_fee_rate=config.taker_fee_rate,
                slippage_bps=config.slippage_bps,
            ),
            window_start=window.test_start,
            window_end=window.test_end,
            prediction_map=test_predictions,
            threshold=threshold,
        )

        baseline_cash = baseline_test.result.final_equity
        ml_cash = ml_test.result.final_equity

        baseline_trade_frames.append(_attach_fold_column(baseline_test.result.trades_frame, window.fold_index))
        ml_trade_frames.append(_attach_fold_column(ml_test.result.trades_frame, window.fold_index))
        baseline_equity_frames.append(
            _attach_fold_column(baseline_test.result.equity_curve.reset_index(), window.fold_index)
        )
        ml_equity_frames.append(
            _attach_fold_column(ml_test.result.equity_curve.reset_index(), window.fold_index)
        )

        selected_config_rows.append(
            {
                "fold_index": window.fold_index,
                "selection_mode": selection_summary["selection_mode"],
                "validation_score": round(float(selection_summary["validation_score"]), 8),
                "train_profit_factor": round(float(selection_summary["train_profit_factor"]), 8),
                "validation_trades_count": int(selection_summary["validation_trades_count"]),
                "validation_max_drawdown_pct": round(
                    float(selection_summary["validation_max_drawdown_pct"]), 8
                ),
                **_config_to_dict(selected_config),
            }
        )
        fold_metrics_rows.append(
            {
                "fold_index": window.fold_index,
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "validation_start": window.validation_start.isoformat(),
                "validation_end": window.validation_end.isoformat(),
                "test_start": window.test_start.isoformat(),
                "test_end": window.test_end.isoformat(),
                "selected_threshold": threshold,
                "train_labels": len(train_labels),
                "train_positive_rate": round(
                    float(train_labels["label"].mean()) if not train_labels.empty else 0.0,
                    8,
                ),
                "validation_baseline_return_pct": round(validation_backtest.result.total_return_pct, 8),
                "validation_baseline_profit_factor": round(validation_backtest.result.profit_factor, 8),
                "validation_ml_return_pct": round(validation_ml_result.total_return_pct, 8),
                "validation_ml_profit_factor": round(validation_ml_result.profit_factor, 8),
                "test_baseline_return_pct": round(baseline_test.result.total_return_pct, 8),
                "test_baseline_profit_factor": round(baseline_test.result.profit_factor, 8),
                "test_baseline_trades_count": baseline_test.result.trades_count,
                "test_ml_return_pct": round(ml_test.result.total_return_pct, 8),
                "test_ml_profit_factor": round(ml_test.result.profit_factor, 8),
                "test_ml_trades_count": ml_test.result.trades_count,
                "baseline_test_final_equity": round(baseline_test.result.final_equity, 8),
                "ml_test_final_equity": round(ml_test.result.final_equity, 8),
            }
        )
        feature_importance_rows.extend(
            _feature_importance_rows(model, fold_index=window.fold_index)
        )

    baseline_trades = _concat_frames(baseline_trade_frames)
    ml_trades = _concat_frames(ml_trade_frames)
    baseline_equity = _concat_frames(baseline_equity_frames)
    ml_equity = _concat_frames(ml_equity_frames)
    baseline_monthly_returns = monthly_returns_from_equity(baseline_equity)
    ml_monthly_returns = monthly_returns_from_equity(ml_equity)

    summary = {
        "product_id": config.product_id,
        "csv_range": {
            "start": candles.index.min().isoformat(),
            "end": candles.index.max().isoformat(),
        },
        "walk_forward": {
            **asdict(config),
            "folds": len(windows),
        },
        "baseline_oos_summary": _summarize_oos_run(
            trades_frame=baseline_trades,
            equity_frame=baseline_equity,
            starting_cash=config.starting_cash,
        ),
        "ml_gated_oos_summary": _summarize_oos_run(
            trades_frame=ml_trades,
            equity_frame=ml_equity,
            starting_cash=config.starting_cash,
        ),
    }

    return WalkForwardRun(
        summary=summary,
        fold_metrics=pd.DataFrame(fold_metrics_rows),
        selected_rule_configs=pd.DataFrame(selected_config_rows),
        baseline_oos_trades=baseline_trades,
        ml_gated_oos_trades=ml_trades,
        baseline_oos_equity=baseline_equity,
        ml_gated_oos_equity=ml_equity,
        baseline_monthly_returns=baseline_monthly_returns,
        ml_gated_monthly_returns=ml_monthly_returns,
        ml_feature_importance=pd.DataFrame(feature_importance_rows),
    )


@dataclass
class _WindowBacktest:
    result: AssetBacktestResult
    feature_frame: pd.DataFrame


def _run_window_backtest(
    *,
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    backtest_config: BacktestConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    prediction_map: Optional[dict[str, float]] = None,
    threshold: Optional[float] = None,
) -> _WindowBacktest:
    window_candles = _window_candles(
        candles=candles,
        strategy_config=strategy_config,
        window_start=window_start,
        window_end=window_end,
    )
    feature_frame = build_signal_frame(window_candles, strategy_config)
    gate = None
    if prediction_map is not None and threshold is not None:
        gate = lambda _row, timestamp: prediction_map.get(timestamp.isoformat(), 0.0) >= threshold
    result = run_backtest_for_asset(
        product_id=product_id,
        candles=window_candles,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        granularity=strategy_config.signal_granularity,
        feature_frame=feature_frame,
        entry_gate=gate,
        entry_start_time=window_start,
    )
    return _WindowBacktest(result=result, feature_frame=feature_frame)


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
    earliest_start = _coerce_utc_timestamp(candles.index.min())
    history_start = max(earliest_start, window_start - pd.to_timedelta(warmup_seconds, unit="s"))
    return candles[(candles.index >= history_start) & (candles.index < window_end)].copy()


def _select_rule_config(
    *,
    candles: pd.DataFrame,
    window: WalkForwardWindow,
    strategy_configs: list[BreakoutConfig],
    research_config: WalkForwardConfig,
) -> tuple[BreakoutConfig, dict[str, Any]]:
    best_valid: Optional[dict[str, Any]] = None
    best_fallback: Optional[dict[str, Any]] = None

    for strategy_config in strategy_configs:
        shared_backtest_config = _build_backtest_config(
            starting_cash=research_config.starting_cash,
            leverage=research_config.leverage,
            taker_fee_rate=research_config.taker_fee_rate,
            slippage_bps=research_config.slippage_bps,
        )
        train_backtest = _run_window_backtest(
            product_id=research_config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest_config,
            window_start=window.train_start,
            window_end=window.train_end,
        )
        validation_backtest = _run_window_backtest(
            product_id=research_config.product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=shared_backtest_config,
            window_start=window.validation_start,
            window_end=window.validation_end,
        )
        raw_score = _raw_result_score(validation_backtest.result)
        candidate = {
            "config": strategy_config,
            "validation_score": raw_score,
            "train_profit_factor": train_backtest.result.profit_factor,
            "validation_trades_count": validation_backtest.result.trades_count,
            "validation_max_drawdown_pct": validation_backtest.result.max_drawdown_pct,
            "selection_mode": "fallback_unconstrained",
        }
        if best_fallback is None or _rule_candidate_key(candidate) > _rule_candidate_key(best_fallback):
            best_fallback = candidate

        if validation_backtest.result.trades_count < MIN_VALIDATION_TRADES:
            continue
        if validation_backtest.result.max_drawdown_pct > MAX_VALIDATION_DRAWDOWN_PCT:
            continue
        candidate["selection_mode"] = "validation_constrained"
        if best_valid is None or _rule_candidate_key(candidate) > _rule_candidate_key(best_valid):
            best_valid = candidate

    selected = best_valid or best_fallback
    if selected is None:
        raise ValueError("Rule-search did not produce any candidate configurations")
    return selected["config"], selected


def _rule_candidate_key(candidate: dict[str, Any]) -> tuple[float, int, float, float]:
    return (
        float(candidate["validation_score"]),
        int(candidate["validation_trades_count"]),
        float(candidate["train_profit_factor"]),
        -float(candidate["validation_max_drawdown_pct"]),
    )


def _build_backtest_config(
    *,
    starting_cash: float,
    leverage: float,
    taker_fee_rate: float,
    slippage_bps: float,
) -> BacktestConfig:
    return BacktestConfig(
        starting_cash=starting_cash,
        leverage=leverage,
        maker_fee_rate=0.0,
        taker_fee_rate=taker_fee_rate,
        entry_liquidity="taker",
        exit_liquidity="taker",
        slippage_bps=slippage_bps,
    )


def _raw_result_score(result: AssetBacktestResult) -> float:
    return (
        float(result.total_return_pct)
        - 0.5 * float(result.max_drawdown_pct)
        + 10.0 * min(float(result.profit_factor), 3.0)
    )


def build_candidate_feature_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame(columns=list(CANDIDATE_FEATURE_COLUMNS))

    long_mask = (
        feature_frame["long_breakout"]
        & feature_frame["kalman_long_ok"]
        & feature_frame["volatility_ok"]
        & feature_frame["volume_ok"]
    )
    short_mask = (
        feature_frame["short_breakout"]
        & feature_frame["kalman_short_ok"]
        & feature_frame["volatility_ok"]
        & feature_frame["volume_ok"]
    )
    return _candidate_feature_frame_from_masks(
        feature_frame=feature_frame,
        long_mask=long_mask,
        short_mask=short_mask,
    )


def build_variant_candidate_feature_frame(
    feature_frame: pd.DataFrame,
    *,
    strategy_config: BreakoutConfig,
    research_variant: ResearchVariant,
) -> pd.DataFrame:
    if feature_frame.empty:
        return pd.DataFrame(columns=list(CANDIDATE_FEATURE_COLUMNS))
    long_mask = _variant_direction_mask(
        feature_frame=feature_frame,
        direction="LONG",
        strategy_config=strategy_config,
        research_variant=research_variant,
    )
    short_mask = _variant_direction_mask(
        feature_frame=feature_frame,
        direction="SHORT",
        strategy_config=strategy_config,
        research_variant=research_variant,
    )
    return _candidate_feature_frame_from_masks(
        feature_frame=feature_frame,
        long_mask=long_mask,
        short_mask=short_mask,
    )


def build_trade_outcome_frame(
    *,
    trades_frame: pd.DataFrame,
    candidate_features: pd.DataFrame,
) -> pd.DataFrame:
    if trades_frame.empty or candidate_features.empty:
        return pd.DataFrame(columns=["signal_time", "pnl", "label", *candidate_features.columns.drop("signal_time", errors="ignore")])
    outcome_frame = trades_frame[["signal_time", "pnl"]].copy()
    outcome_frame["label"] = (outcome_frame["pnl"] > 0).astype(int)
    merged = outcome_frame.merge(candidate_features, on="signal_time", how="inner")
    return merged.drop_duplicates("signal_time")


def _candidate_feature_frame_from_masks(
    *,
    feature_frame: pd.DataFrame,
    long_mask: pd.Series,
    short_mask: pd.Series,
) -> pd.DataFrame:
    long_mask = long_mask.fillna(False).astype(bool)
    short_mask = short_mask.fillna(False).astype(bool)
    candidates = feature_frame.loc[long_mask | short_mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=list(CANDIDATE_FEATURE_COLUMNS))

    direction_sign = pd.Series(1.0, index=candidates.index)
    direction_sign.loc[short_mask[short_mask].index] = -1.0
    box_range = candidates["box_range"].replace(0.0, pd.NA)
    box_atr = candidates["box_atr"].replace(0.0, pd.NA)
    breakout_boundary = candidates["box_high"].where(direction_sign > 0, candidates["box_low"])

    candidate_features = pd.DataFrame(index=candidates.index)
    candidate_features["signal_time"] = candidates.index.map(lambda timestamp: timestamp.isoformat())
    candidate_features["direction"] = direction_sign.map(lambda value: "LONG" if value > 0 else "SHORT")
    candidate_features["direction_long"] = (direction_sign > 0).astype(int)
    candidate_features["box_range_atr_ratio"] = candidates["box_range"] / box_atr
    candidate_features["volume_ratio"] = candidates["volume"] / candidates["volume_ma"].replace(0.0, pd.NA)
    candidate_features["breakout_distance_norm"] = (
        direction_sign * (candidates["close"] - breakout_boundary) / box_range
    )
    candidate_features["kalman_gap_norm"] = (
        direction_sign * (candidates["close"] - candidates["kalman_state"]) / box_atr
    )
    candidate_features["kalman_slope_norm"] = direction_sign * candidates["kalman_slope"] / box_atr
    candidate_features["candle_range_ratio"] = (candidates["high"] - candidates["low"]) / box_range
    candidate_features["candle_body_ratio"] = (candidates["close"] - candidates["open"]).abs() / box_range
    candidate_features["utc_hour"] = candidates.index.hour.astype(int)
    candidate_features = candidate_features.dropna().reset_index(drop=True)
    return candidate_features


def build_trade_label_frame(
    *,
    trades_frame: pd.DataFrame,
    candidate_features: pd.DataFrame,
) -> pd.DataFrame:
    outcome_frame = build_trade_outcome_frame(
        trades_frame=trades_frame,
        candidate_features=candidate_features,
    )
    if outcome_frame.empty:
        return pd.DataFrame(columns=["signal_time", "label", *ML_FEATURE_COLUMNS])
    return outcome_frame[["signal_time", "label", *ML_FEATURE_COLUMNS]].drop_duplicates("signal_time")


def _variant_direction_mask(
    *,
    feature_frame: pd.DataFrame,
    direction: str,
    strategy_config: BreakoutConfig,
    research_variant: ResearchVariant,
) -> pd.Series:
    box_atr = feature_frame["box_atr"].replace(0.0, pd.NA)
    volume_ma = feature_frame["volume_ma"].replace(0.0, pd.NA)
    box_range = feature_frame["box_range"].replace(0.0, pd.NA)

    if direction == "LONG":
        breakout_mask = feature_frame["long_breakout"].fillna(False).astype(bool)
        kalman_side_ok = feature_frame["close"] > feature_frame["kalman_state"]
        kalman_slope_norm = feature_frame["kalman_slope"] / box_atr
        breakout_distance_norm = (feature_frame["close"] - feature_frame["box_high"]) / box_range
        min_box_atr_ratio = _directional_threshold(
            strategy_config.min_box_atr_ratio,
            research_variant.long_min_box_atr_ratio,
        )
        min_volume_ratio = _directional_threshold(
            strategy_config.min_volume_ratio,
            research_variant.long_min_volume_ratio,
        )
        min_breakout_distance = _directional_threshold(
            research_variant.min_breakout_distance_box_ratio,
            research_variant.long_min_breakout_distance_box_ratio,
        )
        min_kalman_slope = _directional_threshold(
            research_variant.kalman_slope_threshold,
            research_variant.long_kalman_slope_threshold,
        )
    else:
        breakout_mask = feature_frame["short_breakout"].fillna(False).astype(bool)
        kalman_side_ok = feature_frame["close"] < feature_frame["kalman_state"]
        kalman_slope_norm = (-feature_frame["kalman_slope"]) / box_atr
        breakout_distance_norm = (feature_frame["box_low"] - feature_frame["close"]) / box_range
        min_box_atr_ratio = _directional_threshold(
            strategy_config.min_box_atr_ratio,
            research_variant.short_min_box_atr_ratio,
        )
        min_volume_ratio = _directional_threshold(
            strategy_config.min_volume_ratio,
            research_variant.short_min_volume_ratio,
        )
        min_breakout_distance = _directional_threshold(
            research_variant.min_breakout_distance_box_ratio,
            research_variant.short_min_breakout_distance_box_ratio,
        )
        min_kalman_slope = _directional_threshold(
            research_variant.kalman_slope_threshold,
            research_variant.short_kalman_slope_threshold,
        )

    return (
        breakout_mask
        & kalman_side_ok.fillna(False)
        & ((feature_frame["box_range"] >= box_atr * min_box_atr_ratio).fillna(False))
        & ((feature_frame["volume"] >= volume_ma * min_volume_ratio).fillna(False))
        & (breakout_distance_norm.fillna(0.0) >= float(min_breakout_distance))
        & (kalman_slope_norm.fillna(0.0) >= max(float(min_kalman_slope), 0.0))
    )


def _directional_threshold(base_value: float, override_value: Optional[float]) -> float:
    return float(override_value if override_value is not None else base_value)


def fit_xgboost_model(train_labels: pd.DataFrame) -> Any:
    if train_labels.empty:
        return ConstantProbabilityModel(0.0)
    positive_probability = float(train_labels["label"].mean())
    if train_labels["label"].nunique() < 2:
        return ConstantProbabilityModel(positive_probability)
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required for the walk-forward ML filter. Install requirements.txt first."
        )
    model = XGBClassifier(
        objective="binary:logistic",
        max_depth=3,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(train_labels[list(ML_FEATURE_COLUMNS)], train_labels["label"])
    return model


def predict_candidate_probabilities(model: Any, candidate_features: pd.DataFrame) -> dict[str, float]:
    if candidate_features.empty:
        return {}
    probabilities = model.predict_proba(candidate_features[list(ML_FEATURE_COLUMNS)])
    return {
        str(signal_time): float(probability[1])
        for signal_time, probability in zip(
            candidate_features["signal_time"].tolist(),
            probabilities,
        )
    }


def _select_threshold(
    *,
    product_id: str,
    candles: pd.DataFrame,
    strategy_config: BreakoutConfig,
    backtest_config: BacktestConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    prediction_map: dict[str, float],
) -> tuple[float, AssetBacktestResult]:
    best_threshold = THRESHOLD_GRID[0]
    best_result: Optional[AssetBacktestResult] = None
    best_key: Optional[tuple[float, int, float]] = None
    for threshold in THRESHOLD_GRID:
        candidate = _run_window_backtest(
            product_id=product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            window_start=window_start,
            window_end=window_end,
            prediction_map=prediction_map,
            threshold=threshold,
        ).result
        candidate_key = (
            _raw_result_score(candidate),
            candidate.trades_count,
            -threshold,
        )
        if best_result is None or candidate_key > best_key:
            best_threshold = threshold
            best_result = candidate
            best_key = candidate_key
    assert best_result is not None
    return best_threshold, best_result


def monthly_returns_from_equity(equity_frame: pd.DataFrame) -> pd.DataFrame:
    if equity_frame.empty:
        return pd.DataFrame(
            columns=[
                "Period",
                "StartTimestamp",
                "EndTimestamp",
                "start_equity",
                "end_equity",
                "return_pct",
            ]
        )

    frame = equity_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp")
    frame["period"] = frame["timestamp"].dt.tz_localize(None).dt.to_period("M")
    rows: list[dict[str, Any]] = []
    for period, period_frame in frame.groupby("period"):
        start_equity = float(period_frame.iloc[0]["equity"])
        end_equity = float(period_frame.iloc[-1]["equity"])
        return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity else 0.0
        rows.append(
            {
                "Period": str(period),
                "StartTimestamp": period_frame.iloc[0]["timestamp"].isoformat(),
                "EndTimestamp": period_frame.iloc[-1]["timestamp"].isoformat(),
                "start_equity": round(start_equity, 8),
                "end_equity": round(end_equity, 8),
                "return_pct": round(return_pct, 8),
            }
        )
    return pd.DataFrame(rows)


def _summarize_oos_run(
    *,
    trades_frame: pd.DataFrame,
    equity_frame: pd.DataFrame,
    starting_cash: float,
) -> dict[str, Any]:
    final_equity = float(equity_frame.iloc[-1]["equity"]) if not equity_frame.empty else starting_cash
    total_return_pct = ((final_equity / starting_cash) - 1.0) * 100.0 if starting_cash else 0.0
    win_rate_pct, profit_factor = _trade_stats(trades_frame)
    max_drawdown_pct = _max_drawdown_from_equity(equity_frame)
    fees_paid = 0.0
    if not trades_frame.empty:
        fees_paid = float(trades_frame["entry_fee"].sum()) + float(trades_frame["exit_fee"].sum())
    return {
        "starting_cash": round(starting_cash, 2),
        "final_equity": round(final_equity, 2),
        "total_return_pct": round(total_return_pct, 4),
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "trades_count": int(len(trades_frame)),
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": round(profit_factor, 4),
        "fees_paid": round(fees_paid, 4),
    }


def _trade_stats(trades_frame: pd.DataFrame) -> tuple[float, float]:
    if trades_frame.empty:
        return 0.0, 0.0
    wins = trades_frame[trades_frame["pnl"] > 0]
    losses = trades_frame[trades_frame["pnl"] < 0]
    win_rate_pct = (len(wins) / len(trades_frame)) * 100.0
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = abs(float(losses["pnl"].sum())) if not losses.empty else 0.0
    if gross_loss == 0.0:
        profit_factor = gross_profit if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    return win_rate_pct, profit_factor


def _max_drawdown_from_equity(equity_frame: pd.DataFrame) -> float:
    if equity_frame.empty:
        return 0.0
    equity_series = equity_frame["equity"].astype(float)
    running_peak = equity_series.cummax()
    drawdown = equity_series / running_peak - 1.0
    return abs(float(drawdown.min()) * 100.0)


def _feature_importance_rows(model: Any, *, fold_index: int) -> list[dict[str, Any]]:
    raw_importances = getattr(model, "feature_importances_", [0.0] * len(ML_FEATURE_COLUMNS))
    rows: list[dict[str, Any]] = []
    for rank, (feature, importance) in enumerate(
        sorted(
            zip(ML_FEATURE_COLUMNS, raw_importances),
            key=lambda item: float(item[1]),
            reverse=True,
        ),
        start=1,
    ):
        rows.append(
            {
                "fold_index": fold_index,
                "feature": feature,
                "importance": round(float(importance), 8),
                "rank": rank,
            }
        )
    return rows


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
        combined["entry_time"] = pd.to_datetime(combined["entry_time"], utc=True)
        combined["exit_time"] = pd.to_datetime(combined["exit_time"], utc=True)
        combined["signal_time"] = pd.to_datetime(combined["signal_time"], utc=True)
        combined = combined.sort_values("entry_time").reset_index(drop=True)
        combined["signal_time"] = combined["signal_time"].map(lambda value: value.isoformat())
        combined["entry_time"] = combined["entry_time"].map(lambda value: value.isoformat())
        combined["exit_time"] = combined["exit_time"].map(lambda value: value.isoformat())
    return combined


def _month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=timestamp.year, month=timestamp.month, day=1, tz="UTC")


def _next_month_start(timestamp: pd.Timestamp) -> pd.Timestamp:
    month_start = _month_start(timestamp)
    if timestamp == month_start:
        return month_start
    return month_start + pd.DateOffset(months=1)


def _coerce_utc_timestamp(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _config_to_dict(config: BreakoutConfig) -> dict[str, Any]:
    normalized = normalize_config(config)
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


__all__ = [
    "CANDIDATE_FEATURE_COLUMNS",
    "MIN_VALIDATION_TRADES",
    "MAX_VALIDATION_DRAWDOWN_PCT",
    "ML_FEATURE_COLUMNS",
    "RULE_SEARCH_SPACE",
    "THRESHOLD_GRID",
    "WalkForwardConfig",
    "WalkForwardRun",
    "WalkForwardWindow",
    "build_candidate_feature_frame",
    "build_trade_outcome_frame",
    "build_trade_label_frame",
    "build_variant_candidate_feature_frame",
    "default_btc_csv_path",
    "fit_xgboost_model",
    "generate_walk_forward_windows",
    "load_btc_candles",
    "monthly_returns_from_equity",
    "predict_candidate_probabilities",
    "run_walk_forward_research",
    "sample_rule_configs",
]
