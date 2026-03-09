# scalping_5min_momentum

Coinbase Advanced Trade momentum scalper for 5-minute or 1-minute spot execution.

## What It Includes

- authenticated Coinbase REST client for candles, product rules, balances, and orders
- EMA/RSI/VWAP/ATR momentum strategy
- paper-trading and live-trading runner
- persistent local JSON state for open position tracking

## Files

- `coinbase_advanced.py`: Coinbase Advanced Trade REST client
- `scalping_strategy.py`: indicator calculation and signal logic
- `run_coinbase_scalper.py`: executable entrypoint
- `requirements.txt`: Python dependencies
- `.gitignore`: local repo hygiene

## Strategy Model

This implementation assumes a long-only spot momentum scalp:

- buy when price is above fast EMA, above slow EMA, above VWAP, RSI is strong, and volume is not weak
- size positions from available quote balance with configurable caps
- set ATR-based stop-loss and take-profit levels
- trail stops upward while the trade remains open
- exit on stop, target, RSI fade, or trend breakdown

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
  --product-id BTC-USD `
  --granularity FIVE_MINUTE `
  --state-path scalping_5min_momentum\btc_paper_state.json
```

## Live Trading

```powershell
py scalping_5min_momentum\run_coinbase_scalper.py `
  --mode live `
  --product-id BTC-USD `
  --granularity FIVE_MINUTE `
  --quote-size 25 `
  --state-path scalping_5min_momentum\btc_live_state.json
```

## Notes

- `paper` mode updates a simulated wallet in the state file.
- `live` mode uses Coinbase order preview before order submission.
- Default parameters are configurable through CLI flags in `run_coinbase_scalper.py`.
- The strategy was implemented from an explicit momentum-scalping assumption because the PDF rules were not machine-readable in this shell.
