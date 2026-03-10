# Back Testing

This folder contains a historical simulator for the `scalping_5min_momentum` strategy.

## Default Market Set

- `BTC-PERP`
- `ETH-PERP`
- `SOL-PERP`
- `XRP-PERP`

## What It Does

- downloads historical Coinbase perpetual candles
- caches each request to CSV under `back_testing/data`
- runs the momentum strategy on each market independently
- writes trades, equity curves, and a summary under `back_testing/output`
- can query current maker/taker fee rates from Coinbase via the Advanced Trade API
- supports configurable margin allocation and leverage in the simulator
- includes a reusable PowerShell downloader for long-range historical CSV exports

## Key Assumptions

- long-only strategy
- signal is generated on bar close
- close-based signal orders execute on the next bar open
- futures sizing uses margin allocation multiplied by leverage to determine notional exposure
- ATR stop-loss and take-profit are checked against candle low/high
- stop-loss wins if stop and target are both hit in the same candle
- liquidation is modeled approximately from entry price and leverage
- fees and slippage are configurable

## Coinbase Constraints

- Coinbase help currently describes perpetual futures leverage up to `50x` on eligible contracts.
- If you run the simulator above `50x`, that result is hypothetical rather than Coinbase-faithful.

## Example

```powershell
py scalping_5min_momentum\back_testing\run_backtest.py `
  --products BTC-PERP ETH-PERP SOL-PERP XRP-PERP `
  --granularity FIVE_MINUTE `
  --days 90 `
  --starting-cash 10000 `
  --position-allocation 0.81 `
  --leverage 50 `
  --slippage-bps 2
```

## Outputs

- `back_testing/output/backtest_summary.json`
- `back_testing/output/asset_summaries.csv`
- `back_testing/output/<PRODUCT>_trades.csv`
- `back_testing/output/<PRODUCT>_equity.csv`

## Historical Data

See `back_testing/DATA_DOWNLOAD.md` and `back_testing/download_coinbase_candles.ps1` for raw CSV downloads by product and date range.
