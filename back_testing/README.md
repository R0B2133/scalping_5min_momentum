# Back Testing

This folder contains the historical simulator for the perps box-breakout strategy.

## Default Market Set

- `BTC-PERP`
- `ETH-PERP`
- `SOL-PERP`
- `XRP-PERP`

## What It Does

- downloads historical Coinbase perpetual candles using the signal timeframe
- caches each request to CSV under `back_testing/data`
- runs the breakout strategy on each market independently
- supports both `strict_1m_on_5m` and `5m_only` modes
- models long and short positions, leverage, fees, slippage, and approximate liquidation
- writes trades, equity curves, and a summary under `back_testing/output`

## Key Assumptions

- signals are generated on bar close
- entry orders fill at the next bar open
- stop-loss and take-profit are checked against candle high/low intrabar
- stop-loss wins if stop and target are both hit inside the same candle
- sizing is based on configured account risk and stop distance, then capped by leverage and notional limits
- one position per symbol is open at a time

## Coinbase Constraints

- leverage is intentionally capped at `10x` in the Python entrypoints
- product ids may be passed as aliases like `BTC-PERP`, with client-side normalization for live API calls

## Example

```powershell
py scalping_5min_momentum\back_testing\run_backtest.py `
  --products BTC-PERP ETH-PERP SOL-PERP XRP-PERP `
  --timeframe-mode strict_1m_on_5m `
  --signal-granularity ONE_MINUTE `
  --context-granularity FIVE_MINUTE `
  --days 90 `
  --starting-cash 10000 `
  --leverage 2 `
  --slippage-bps 2
```

## BTC Optimization Sequence

The BTC research runner now implements the one-change-at-a-time optimization roadmap.

Default mode runs the full sequence:

1. side filter
2. context timeframe sweep
3. trade suppression
4. entry quality tightening
5. stop/target redesign
6. execution sensitivity

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode optimization_sequence `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --train-months 6 `
  --validation-months 1 `
  --test-months 1 `
  --output-dir output_btc_optimization_taker_taker
```

To evaluate a single research variant directly:

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode single_variant `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --side-mode long_only `
  --blocked-utc-hours 2 3 4 5 `
  --cooldown-minutes 5 `
  --one-trade-per-box `
  --long-min-box-atr-ratio 0.9 `
  --long-min-volume-ratio 1.05 `
  --output-dir output_btc_long_only_variant
```

Single-variant mode accepts the research controls introduced by the roadmap:

- `--context-granularity`
- `--side-mode`
- `--blocked-utc-hours`
- `--cooldown-minutes`
- `--one-trade-per-box`
- `--min-breakout-distance-box-ratio`
- `--kalman-slope-threshold`
- `--long-kalman-slope-threshold`
- `--short-kalman-slope-threshold`
- `--stop-family`
- `--stop-buffer-atr`
- `--long-reward-risk-ratio`
- `--short-reward-risk-ratio`
- `--long-min-box-atr-ratio`
- `--short-min-box-atr-ratio`
- `--long-min-volume-ratio`
- `--short-min-volume-ratio`
- `--long-min-breakout-distance-box-ratio`
- `--short-min-breakout-distance-box-ratio`
- `--time-stop-bars`
- `--breakeven-trigger-r`

Optimization-sequence mode also exposes the search grids directly:

- `--context-granularities`
- `--side-modes`
- `--stop-families`
- `--stop-buffer-atr-grid`
- `--long-reward-risk-grid`
- `--short-reward-risk-grid`
- `--min-box-atr-grid`
- `--min-volume-ratio-grid`
- `--breakout-distance-grid`
- `--kalman-slope-threshold-grid`
- `--blocked-hour-counts`
- `--cooldown-grid`
- `--one-trade-per-box-options`
- `--time-stop-grid`
- `--breakeven-grid`

To run the later ML layer explicitly and separately from the default rule-first workflow:

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode xgboost_filter_research `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --context-granularity FIFTEEN_MINUTE `
  --side-mode long_only `
  --output-dir output_btc_xgboost_filter
```

To learn an adaptive breakout gate from train-fold trade history, using volume plus breakout and box-quality regimes instead of one fixed threshold:

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode regime_filter_research `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --context-granularity FIFTEEN_MINUTE `
  --side-mode long_only `
  --regime-min-samples 8 `
  --regime-min-profit-factor 1.0 `
  --output-dir output_btc_regime_filter
```

## Outputs

- `back_testing/output/backtest_summary.json`
- `back_testing/output/asset_summaries.csv`
- `back_testing/output/<PRODUCT>_trades.csv`
- `back_testing/output/<PRODUCT>_equity.csv`
- `back_testing/output_btc_optimization_taker_taker/optimization_summary.json`
- `back_testing/output_btc_optimization_taker_taker/experiment_comparison.csv`
- `back_testing/output_btc_optimization_taker_taker/best_variant_fold_metrics.csv`
- `back_testing/output_btc_optimization_taker_taker/best_variant_trades.csv`
- `back_testing/output_btc_optimization_taker_taker/best_variant_equity.csv`
- `back_testing/output_btc_optimization_taker_taker/best_variant_monthly_returns.csv`
- `back_testing/<custom_output_dir>/variant_summary.json`
- `back_testing/<custom_output_dir>/variant_fold_metrics.csv`
- `back_testing/<custom_output_dir>/variant_trades.csv`
- `back_testing/<custom_output_dir>/variant_equity.csv`
- `back_testing/<custom_output_dir>/variant_monthly_returns.csv`
- `back_testing/<custom_output_dir>/xgboost_filter_summary.json`
- `back_testing/<custom_output_dir>/baseline_fold_metrics.csv`
- `back_testing/<custom_output_dir>/xgboost_fold_metrics.csv`
- `back_testing/<custom_output_dir>/regime_filter_summary.json`
- `back_testing/<custom_output_dir>/regime_fold_metrics.csv`
- `back_testing/<custom_output_dir>/regime_filtered_trades.csv`
- `back_testing/<custom_output_dir>/regime_filtered_equity.csv`
- `back_testing/<custom_output_dir>/regime_filtered_monthly_returns.csv`
- `back_testing/<custom_output_dir>/train_regime_table.csv`

## Historical Data

See `back_testing/DATA_DOWNLOAD.md` and `back_testing/download_coinbase_candles.ps1` for raw CSV downloads by product and date range.

## ML Dependency

The default optimization sequence is rule-only. The new `regime_filter_research` mode is also rule-based and learns allowed volume/breakout/box regimes from train-fold trade history. `xgboost` is only needed for the separate `xgboost_filter_research` mode. If `xgboost` is not installed in the active Python runtime, that explicit ML mode is skipped and the skip reason is recorded in the output summary.
