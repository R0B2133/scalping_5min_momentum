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

## Outputs

- `back_testing/output/backtest_summary.json`
- `back_testing/output/asset_summaries.csv`
- `back_testing/output/<PRODUCT>_trades.csv`
- `back_testing/output/<PRODUCT>_equity.csv`

## Historical Data

See `back_testing/DATA_DOWNLOAD.md` and `back_testing/download_coinbase_candles.ps1` for raw CSV downloads by product and date range.
