# Current Strategy Status

This document captures the state of the `scalping_5min_momentum` project as of March 17, 2026.

## Status

The strategy is being postponed for now.

The codebase contains a working breakout research and execution framework, but the current strategy family is not yet profitable enough to justify live deployment. Research improved the results a lot compared with the earliest `1m/5m` baseline, but the best cross-asset variants still remain slightly negative after realistic fees and slippage.

## Current Strategy Design

The project no longer uses the old EMA/RSI/VWAP momentum model. The current logic is a box-breakout system for Coinbase perpetual futures:

- build a box from the previous completed context candle
- wait for price to break above the box high or below the box low
- require price, volume, and Kalman trend alignment before entry
- size positions from risk budget, leverage, and exchange constraints
- place stops from market structure rather than wide trailing logic
- manage one position per symbol at a time

The research stack now supports slower structures beyond classic scalping:

- configurable signal bars, including `ONE_MINUTE`, `THIRTY_MINUTE`, and `ONE_HOUR`
- configurable context boxes, including `FIVE_MINUTE`, `ONE_HOUR`, and `FOUR_HOUR`
- side filtering: `both`, `long_only`, `short_only`
- one-trade-per-box throttling
- stop families such as `breakout_candle` and `box_edge`
- reward/risk sweeps instead of fixed `1:1.5`

## What Was Learned

### 1. The original fast baseline was poor

The raw `1m` signal against `5m` boxes overtraded badly. On both BTC and ETH, that baseline produced very high trade counts, low win rates, weak profit factors, and near-complete capital loss over full-history tests under `taker/taker` fees.

### 2. Slower structures helped materially

The strongest improvements came from moving away from `1m/5m` behavior and into slower breakout structures such as:

- `THIRTY_MINUTE` signal with `ONE_HOUR` context
- `ONE_HOUR` signal with `FOUR_HOUR` context

These changes reduced overtrading, improved win rate, and brought drawdown down sharply.

### 3. Long-only variants were more stable

In the better ETH cross-tests, `long_only` variants consistently behaved better than symmetric long/short variants.

### 4. Tight filters improved quality

The more promising variants used tighter filters such as:

- `min_box_atr_ratio` around `1.4` to `1.6`
- `min_volume_ratio` around `1.2` to `1.3`
- `one_trade_per_box = true`

## Best Results So Far

### Best slower-structure cross-asset result

BTC-trained, tested on full-history ETH:

- `signal_granularity = THIRTY_MINUTE`
- `context_granularity = ONE_HOUR`
- `side_mode = long_only`
- `reward_risk_ratio = 1.25`
- `min_box_atr_ratio = 1.4`
- `min_volume_ratio = 1.2`
- `one_trade_per_box = true`

ETH result:

- return: `-3.2368%`
- max drawdown: `3.4004%`
- win rate: `42.9%`
- profit factor: `0.7215`
- trades: `331`

### Best win-rate-focused variant

BTC-trained, tested on full-history ETH:

- `signal_granularity = THIRTY_MINUTE`
- `context_granularity = ONE_HOUR`
- `side_mode = long_only`
- `reward_risk_ratio = 0.75`
- `stop_family = box_edge`
- `min_box_atr_ratio = 1.6`
- `min_volume_ratio = 1.3`
- `one_trade_per_box = true`

ETH result:

- return: `-0.6120%`
- max drawdown: `2.3421%`
- win rate: `59.13%`
- profit factor: `0.9497`
- trades: `208`

### Best balanced trade-off so far

BTC-trained, tested on full-history ETH:

- `signal_granularity = ONE_HOUR`
- `context_granularity = FOUR_HOUR`
- `side_mode = long_only`
- `reward_risk_ratio = 1.0`
- `stop_family = breakout_candle`
- `min_box_atr_ratio = 1.6`
- `min_volume_ratio = 1.3`
- `one_trade_per_box = true`

ETH result:

- return: `-0.1680%`
- max drawdown: `1.0465%`
- win rate: `52.83%`
- profit factor: `0.9598`
- trades: `106`
- final equity from `$10,000`: `$9,983.20`

This is the strongest result achieved so far. It is close to flat and much less fragile than the early versions, but it is still not profitable enough to treat as production-ready.

## Why The Strategy Is Being Postponed

Even after meaningful improvement:

- return is still slightly negative
- profit factor is still below `1.0`
- the current edge is too thin to trust after costs
- the better versions look more like slower breakout trading than the original short-horizon scalping idea

In other words, the framework is useful, but this exact strategy family has not yet earned deployment.

## Codebase Value Kept

The project still contains useful building blocks:

- Coinbase Advanced Trade perp client
- backtest engine with fees, slippage, leverage, and rule variants
- walk-forward research runner
- BTC and ETH local research datasets
- rule-first optimization framework

That means the work is not wasted. The project can be resumed later from a much stronger foundation.

## If Research Resumes Later

Recommended next directions:

- keep focusing on slower structures such as `1H/4H`
- continue with rule-first filters before reintroducing ML
- use XGBoost only as a later meta-filter on top of a profitable rule baseline
- test time-of-day/session filters
- test more structure-aware exits and holding-time rules
- validate on BTC, ETH, SOL, and XRP once more perp data is collected

## Reference Outputs

The key research artifacts referenced in this summary were produced locally in these folders:

- `back_testing/output_btc_windows_train_eth_full_multisignal_refine_20260317`
- `back_testing/output_btc_train_eth_winrate_20260317`
- `back_testing/output_btc_train_eth_balanced_20260317`

These are local research outputs and are not required for the source repository itself.
