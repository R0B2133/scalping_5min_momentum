# scalping_5min_momentum

Coinbase perpetual futures box-breakout scalper built from the PDF strategy outline.

Current project status: the strategy is being postponed for now. A concise research summary and the latest best-known results are documented in [CURRENT_STRATEGY_STATUS.md](CURRENT_STRATEGY_STATUS.md).

## What It Includes

- authenticated Coinbase Advanced Trade REST client for candles, fee tiers, product rules, and orders
- rule-based breakout strategy with `strict_1m_on_5m` and `5m_only` modes
- Kalman trend-bias filter plus volatility and volume filters
- risk-based position sizing with configurable leverage up to `10x`
- continuous paper/live runner with schema-versioned JSON state
- multi-asset backtesting for Coinbase perpetual futures

## Strategy Model

This package no longer uses the old EMA/RSI/VWAP momentum model.

The current strategy is a box-breakout framework:

- build the active box from the previous completed context range
- trade breakouts above the box high or below the box low
- use the breakout candle's opposite extreme as the stop reference
- set the take-profit from a configurable reward/risk ratio, default `1.5`
- require Kalman trend bias, minimum box size versus ATR, and non-weak volume
- allow only one open position per symbol at a time

## Files

- `coinbase_advanced.py`: Coinbase Advanced Trade REST client with perps-friendly helpers
- `scalping_strategy.py`: breakout signal engine, Kalman filter, and sizing logic
- `run_coinbase_scalper.py`: continuous paper/live runner for perps
- `back_testing/`: historical simulator for the same breakout model
- `requirements.txt`: runtime dependencies

## Credentials

Place your Coinbase CDP API key file at:

```text
quant-lab/cdp_api_key.json
```

Or set:

```text
COINBASE_KEY_NAME
COINBASE_PRIVATE_KEY
```

## Install

```powershell
pip install -r scalping_5min_momentum\requirements.txt
```

## Paper Trading

```powershell
py scalping_5min_momentum\run_coinbase_scalper.py `
  --mode paper `
  --product-id BTC-PERP `
  --timeframe-mode strict_1m_on_5m `
  --signal-granularity ONE_MINUTE `
  --context-granularity FIVE_MINUTE `
  --state-path scalping_5min_momentum\btc_perp_paper_state.json
```

## Live Trading

```powershell
py scalping_5min_momentum\run_coinbase_scalper.py `
  --mode live `
  --product-id BTC-PERP `
  --leverage 2 `
  --margin-type CROSS `
  --state-path scalping_5min_momentum\btc_perp_live_state.json
```

## Live Dry Run

```powershell
py scalping_5min_momentum\run_coinbase_scalper.py `
  --mode live `
  --dry-run `
  --max-iterations 1 `
  --product-id BTC-PERP
```

## BTC Optimization Research

Use the saved local `BTC-PERP` minute CSV to run the step-by-step BTC optimization roadmap:

- compare `both`, `long_only`, and `short_only`
- sweep context boxes across `FIVE_MINUTE`, `FIFTEEN_MINUTE`, and `THIRTY_MINUTE`
- block the worst UTC trading hours
- test cooldowns and `one_trade_per_box`
- tighten box, volume, breakout-distance, and Kalman slope thresholds
- test stop families plus asymmetric reward/risk, time stops, and breakeven rules
- validate execution sensitivity with `taker/taker` and `maker/taker`

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode optimization_sequence `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --output-dir output_btc_optimization_taker_taker
```

To evaluate one explicit research variant instead of the full sequence:

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode single_variant `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --side-mode long_only `
  --blocked-utc-hours 2 3 4 5 `
  --cooldown-minutes 5 `
  --one-trade-per-box `
  --output-dir output_btc_long_only_variant
```

To run the ML layer separately after a rule-based candidate is stable:

```powershell
py scalping_5min_momentum\back_testing\run_walk_forward.py `
  --mode xgboost_filter_research `
  --csv-path scalping_5min_momentum\back_testing\output_local\maker_maker\BTC_PERP_INTX_ONE_MINUTE_20230830_20260316.csv `
  --product-id BTC-PERP `
  --context-granularity FIFTEEN_MINUTE `
  --side-mode long_only `
  --output-dir output_btc_xgboost_filter
```

To learn a rule-based adaptive gate from train-fold breakout history, using volume with breakout and box-quality regimes instead of one fixed threshold:

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

## Notes

- `paper` mode tracks realized PnL, fees, and mark-to-market equity in the state file.
- `live` mode uses order preview before submission and can validate without placing orders via `--dry-run`.
- old state files from the momentum implementation are rejected by schema version.
- `back_testing` reuses the same breakout logic, with next-bar entry fills and intrabar stop/target checks.
- `run_walk_forward.py` now defaults to a rule-only optimization sequence and exposes timeframe, stop-family, Kalman-slope, and exit grids directly.
- `run_walk_forward.py` supports `optimization_sequence`, `single_variant`, `regime_filter_research`, and `xgboost_filter_research` modes.
- `regime_filter_research` learns allowed breakout regimes from train-fold price/volume history and applies them out of sample through the same backtest engine.
- the XGBoost layer requires `xgboost`; if it is not installed, the separate ML mode is skipped gracefully and the reason is recorded in the output summary.
