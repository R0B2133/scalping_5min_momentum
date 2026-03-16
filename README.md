# scalping_5min_momentum

Coinbase perpetual futures box-breakout scalper built from the PDF strategy outline.

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

## Notes

- `paper` mode tracks realized PnL, fees, and mark-to-market equity in the state file.
- `live` mode uses order preview before submission and can validate without placing orders via `--dry-run`.
- old state files from the momentum implementation are rejected by schema version.
- `back_testing` reuses the same breakout logic, with next-bar entry fills and intrabar stop/target checks.
