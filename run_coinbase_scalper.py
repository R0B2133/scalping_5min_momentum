from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coinbase_advanced import (
    CoinbaseAdvancedClient,
    GRANULARITY_TO_SECONDS,
    round_to_increment,
)
from scalping_strategy import (
    PositionState,
    ScalpingConfig,
    evaluate_scalping_decision,
    required_history_bars,
    suggest_quote_order_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Coinbase Advanced Trade scalping strategy in paper or live mode.",
    )
    parser.add_argument("--product-id", default="BTC-USD")
    parser.add_argument("--granularity", default="ONE_MINUTE", choices=sorted(GRANULARITY_TO_SECONDS))
    parser.add_argument("--lookback-bars", type=int, default=240)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--credentials-path", default=None)
    parser.add_argument("--state-path", default="scalper_state.json")
    parser.add_argument("--quote-size", type=float, default=None)
    parser.add_argument("--paper-cash", type=float, default=1000.0)
    parser.add_argument("--fast-ema", type=int, default=8)
    parser.add_argument("--slow-ema", type=int, default=21)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--min-rsi-entry", type=float, default=55.0)
    parser.add_argument("--max-rsi-entry", type=float, default=78.0)
    parser.add_argument("--exit-rsi", type=float, default=48.0)
    parser.add_argument("--stop-atr", type=float, default=1.2)
    parser.add_argument("--take-profit-atr", type=float, default=1.8)
    parser.add_argument("--trailing-atr", type=float, default=1.0)
    parser.add_argument("--risk-fraction", type=float, default=0.02)
    parser.add_argument("--min-notional", type=float, default=10.0)
    parser.add_argument("--max-notional", type=float, default=100.0)
    return parser.parse_args()


def load_state(state_path: Path, product_id: str, paper_cash: float) -> dict[str, Any]:
    base_currency, quote_currency = product_id.split("-", maxsplit=1)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        existing_product_id = state.get("product_id")
        if existing_product_id and existing_product_id != product_id:
            raise ValueError(
                f"State file {state_path} belongs to {existing_product_id}, not {product_id}"
            )
        return state
    return {
        "product_id": product_id,
        "position": asdict(PositionState()),
        "paper_wallet": {
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "base_balance": 0.0,
            "cash_balance": paper_cash,
            "equity": paper_cash,
        },
        "last_decision": None,
        "last_execution": None,
        "updated_at": None,
    }


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def position_from_state(state: dict[str, Any]) -> PositionState:
    return PositionState(**state.get("position", {}))


def build_config(args: argparse.Namespace) -> ScalpingConfig:
    return ScalpingConfig(
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        rsi_period=args.rsi_period,
        atr_period=args.atr_period,
        volume_window=args.volume_window,
        min_rsi_entry=args.min_rsi_entry,
        max_rsi_entry=args.max_rsi_entry,
        exit_rsi=args.exit_rsi,
        stop_atr_multiple=args.stop_atr,
        take_profit_atr_multiple=args.take_profit_atr,
        trailing_atr_multiple=args.trailing_atr,
        risk_fraction=args.risk_fraction,
        min_quote_notional=args.min_notional,
        max_quote_notional=args.max_notional,
    )


def parse_product_rules(product: dict[str, Any]) -> dict[str, float]:
    return {
        "base_increment": float(product.get("base_increment", "0.00000001")),
        "quote_increment": float(product.get("quote_increment", "0.01")),
        "base_min_size": float(product.get("base_min_size", "0.0")),
        "quote_min_size": float(product.get("quote_min_size", "0.0")),
    }


def resolve_quote_size(
    args: argparse.Namespace,
    config: ScalpingConfig,
    available_quote_balance: float,
    product_rules: dict[str, float],
) -> float:
    target = args.quote_size
    if target is None:
        target = suggest_quote_order_size(
            available_quote_balance=available_quote_balance,
            config=config,
            quote_min_size=product_rules["quote_min_size"],
        )
    return round_to_increment(target, product_rules["quote_increment"])


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    config = build_config(args)
    client = CoinbaseAdvancedClient(credentials_path=args.credentials_path)
    state = load_state(state_path, args.product_id, args.paper_cash)
    position = position_from_state(state)

    history_bars = max(args.lookback_bars, required_history_bars(config) + 5)
    now = datetime.now(timezone.utc)
    granularity_seconds = GRANULARITY_TO_SECONDS[args.granularity]
    end_time = int(now.timestamp())
    start_time = end_time - history_bars * granularity_seconds

    candles = client.fetch_candles(
        product_id=args.product_id,
        start_time=start_time,
        end_time=end_time,
        granularity=args.granularity,
    )
    if candles.empty:
        raise RuntimeError(f"No candles returned for {args.product_id}")

    product = client.get_product(args.product_id)
    product_rules = parse_product_rules(product)
    best_bid_ask = client.get_best_bid_ask([args.product_id])
    top_of_book = client.get_top_of_book(best_bid_ask, args.product_id)

    if args.mode == "paper":
        available_quote_balance = float(state["paper_wallet"]["cash_balance"])
    else:
        quote_currency = args.product_id.split("-", maxsplit=1)[1]
        live_balances = client.get_available_balances()
        available_quote_balance = float(live_balances.get(quote_currency, 0.0))

    decision = evaluate_scalping_decision(
        candles=candles,
        config=config,
        position=position,
        best_bid=top_of_book["bid"],
        best_ask=top_of_book["ask"],
    )

    execution: dict[str, Any] | None = None
    timestamp = now.isoformat()
    if decision.action == "BUY":
        quote_size = resolve_quote_size(args, config, available_quote_balance, product_rules)
        if quote_size < product_rules["quote_min_size"]:
            execution = {
                "action": "SKIP",
                "reason": "Quote size below Coinbase minimum",
                "quote_size": quote_size,
            }
        else:
            estimated_base_size = round_to_increment(
                quote_size / decision.latest_price,
                product_rules["base_increment"],
            )
            if estimated_base_size < product_rules["base_min_size"]:
                execution = {
                    "action": "SKIP",
                    "reason": "Estimated base size below Coinbase minimum",
                    "base_size": estimated_base_size,
                }
            elif args.mode == "paper":
                spent = estimated_base_size * decision.latest_price
                state["paper_wallet"]["cash_balance"] -= spent
                state["paper_wallet"]["base_balance"] += estimated_base_size
                position = PositionState(
                    is_open=True,
                    entry_price=decision.latest_price,
                    base_size=estimated_base_size,
                    quote_size=spent,
                    stop_price=decision.stop_price,
                    take_profit_price=decision.take_profit_price,
                    opened_at=timestamp,
                    order_id="paper-buy",
                )
                execution = {
                    "action": "BUY",
                    "mode": "paper",
                    "fill_price": decision.latest_price,
                    "base_size": estimated_base_size,
                    "quote_size": spent,
                }
            else:
                preview = client.preview_market_order(
                    product_id=args.product_id,
                    side="BUY",
                    quote_size=quote_size,
                )
                preview_id = preview.get("preview_id")
                order = client.create_market_order(
                    product_id=args.product_id,
                    side="BUY",
                    quote_size=quote_size,
                    preview_id=preview_id,
                )
                order_id = (
                    order.get("success_response", {}).get("order_id")
                    or order.get("order_id")
                    or "submitted"
                )
                preview_base = preview.get("order_total", {}).get("base_size")
                base_size = estimated_base_size
                if preview_base is not None:
                    base_size = round_to_increment(float(preview_base), product_rules["base_increment"])
                position = PositionState(
                    is_open=True,
                    entry_price=decision.latest_price,
                    base_size=base_size,
                    quote_size=quote_size,
                    stop_price=decision.stop_price,
                    take_profit_price=decision.take_profit_price,
                    opened_at=timestamp,
                    order_id=str(order_id),
                )
                execution = {
                    "action": "BUY",
                    "mode": "live",
                    "order_id": order_id,
                    "quote_size": quote_size,
                    "estimated_base_size": base_size,
                }
    elif decision.action == "SELL" and position.is_open:
        base_size = round_to_increment(position.base_size, product_rules["base_increment"])
        if base_size < product_rules["base_min_size"]:
            execution = {
                "action": "SKIP",
                "reason": "Position size below Coinbase minimum",
                "base_size": base_size,
            }
        elif args.mode == "paper":
            proceeds = base_size * decision.latest_price
            state["paper_wallet"]["cash_balance"] += proceeds
            state["paper_wallet"]["base_balance"] = max(
                0.0,
                state["paper_wallet"]["base_balance"] - base_size,
            )
            execution = {
                "action": "SELL",
                "mode": "paper",
                "fill_price": decision.latest_price,
                "base_size": base_size,
                "quote_size": proceeds,
                "pnl": proceeds - position.quote_size,
            }
            position = PositionState()
        else:
            preview = client.preview_market_order(
                product_id=args.product_id,
                side="SELL",
                base_size=base_size,
            )
            preview_id = preview.get("preview_id")
            order = client.create_market_order(
                product_id=args.product_id,
                side="SELL",
                base_size=base_size,
                preview_id=preview_id,
            )
            order_id = (
                order.get("success_response", {}).get("order_id")
                or order.get("order_id")
                or "submitted"
            )
            execution = {
                "action": "SELL",
                "mode": "live",
                "order_id": order_id,
                "base_size": base_size,
            }
            position = PositionState()
    else:
        if position.is_open:
            position.stop_price = decision.stop_price
            position.take_profit_price = decision.take_profit_price
        execution = {
            "action": "HOLD",
            "mode": args.mode,
        }

    state["position"] = asdict(position)
    if args.mode == "paper":
        mark_price = top_of_book["mid"] or candles["close"].iloc[-1]
        state["paper_wallet"]["equity"] = (
            state["paper_wallet"]["cash_balance"] + state["paper_wallet"]["base_balance"] * mark_price
        )
    state["last_decision"] = decision.to_dict()
    state["last_execution"] = execution
    state["updated_at"] = timestamp
    save_state(state_path, state)

    output = {
        "timestamp": timestamp,
        "product_id": args.product_id,
        "mode": args.mode,
        "decision": decision.to_dict(),
        "execution": execution,
        "position": state["position"],
        "paper_wallet": state.get("paper_wallet"),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
