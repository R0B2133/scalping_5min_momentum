# Data Download Outline

Use `download_coinbase_candles.ps1` to fetch historical Coinbase candles as CSV.

For the current breakout strategy, the default research dataset is perpetual futures data at `ONE_MINUTE`,
because the runner/backtest can resample that into the `strict_1m_on_5m` context internally.

## Inputs

- `ProductId`
  - perpetual examples: `BTC-PERP`, `ETH-PERP`, `SOL-PERP`, `XRP-PERP`
- `Granularity`
  - for this strategy: `ONE_MINUTE` for `strict_1m_on_5m`
  - `FIVE_MINUTE` is still valid for `5m_only`
- `StartUtc`
  - ISO-8601 UTC time such as `2016-01-01T00:00:00Z`
- `EndUtc`
  - ISO-8601 UTC time such as `2026-03-08T23:59:00Z`
- `OutputPath`
  - CSV destination

## BTC Perp Example

```powershell
powershell -ExecutionPolicy Bypass -File .\scalping_5min_momentum\back_testing\download_coinbase_candles.ps1 `
  -ProductId BTC-PERP `
  -Granularity ONE_MINUTE `
  -StartUtc 2016-01-01T00:00:00Z `
  -EndUtc 2026-03-08T23:59:00Z `
  -OutputPath D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\BTC_PERP_ONE_MINUTE_20160101_20260308.csv
```

## Other Assets

ETH:

```powershell
powershell -ExecutionPolicy Bypass -File .\scalping_5min_momentum\back_testing\download_coinbase_candles.ps1 `
  -ProductId ETH-PERP `
  -Granularity ONE_MINUTE `
  -StartUtc 2016-01-01T00:00:00Z `
  -EndUtc 2026-03-08T23:59:00Z `
  -OutputPath D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\ETH_PERP_ONE_MINUTE_20160101_20260308.csv
```

SOL:

```powershell
powershell -ExecutionPolicy Bypass -File .\scalping_5min_momentum\back_testing\download_coinbase_candles.ps1 `
  -ProductId SOL-USD `
  -Granularity FIVE_MINUTE `
  -StartUtc 2020-01-01T00:00:00Z `
  -EndUtc 2026-03-08T23:59:00Z `
  -OutputPath D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\SOL_USD_FIVE_MINUTE_20200101_20260308.csv
```

XRP:

```powershell
powershell -ExecutionPolicy Bypass -File .\scalping_5min_momentum\back_testing\download_coinbase_candles.ps1 `
  -ProductId XRP-USD `
  -Granularity FIVE_MINUTE `
  -StartUtc 2016-01-01T00:00:00Z `
  -EndUtc 2026-03-08T23:59:00Z `
  -OutputPath D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\XRP_USD_FIVE_MINUTE_20160101_20260308.csv
```

## Notes

- Perpetual products usually do not have 10 years of history, so use spot pairs for long-span training data.
- The script authenticates with `D:\Quant\quant-lab\cdp_api_key.json` by default.
- Coinbase returns candles in chunks, so long ranges can take thousands of API calls.
- For very long ranges, split the download into yearly or multi-year segments and merge them after completion.
- Use `merge_candle_segments.ps1` to combine segment CSVs into one file with a single header.
- Output columns are:
  - `product_id`
  - `granularity`
  - `timestamp_utc`
  - `start_unix`
  - `open`
  - `high`
  - `low`
  - `close`
  - `volume`

## Long-Range Workflow

1. Choose the spot or perpetual `ProductId`.
2. Choose `StartUtc`, `EndUtc`, and `Granularity`.
3. For large ranges, split the date span into yearly or multi-year segments.
4. Run `download_coinbase_candles.ps1` once per segment.
5. Merge the finished files with `merge_candle_segments.ps1`.
6. Audit the final CSV for missing candles before using it for backtesting or training.

Example merge:

```powershell
powershell -ExecutionPolicy Bypass -File .\scalping_5min_momentum\back_testing\merge_candle_segments.ps1 `
  -InputDirectory D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\btc_segments `
  -OutputPath D:\Quant\quant-lab\scalping_5min_momentum\back_testing\data\BTC_USD_FIVE_MINUTE_20160101_20260308.csv
```
