from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from coinbase_advanced import (
        CoinbaseAdvancedClient,
        GRANULARITY_TO_SECONDS,
        round_to_increment,
    )
    from scalping_strategy import (
        SCHEMA_VERSION,
        BreakoutConfig,
        PositionState,
        calculate_position_size,
        evaluate_live_exit,
        evaluate_signal,
        mark_to_market_pnl,
        normalize_config,
        required_history_bars,
    )
except ImportError:  # pragma: no cover - package import path
    from scalping_5min_momentum.coinbase_advanced import (
        CoinbaseAdvancedClient,
        GRANULARITY_TO_SECONDS,
        round_to_increment,
    )
    from scalping_5min_momentum.scalping_strategy import (
        SCHEMA_VERSION,
        BreakoutConfig,
        PositionState,
        calculate_position_size,
        evaluate_live_exit,
        evaluate_signal,
        mark_to_market_pnl,
        normalize_config,
        required_history_bars,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the perps box-breakout scalping strategy in paper or live mode.",
    )
    parser.add_argument("--product-id", default="BTC-PERP")
    parser.add_argument("--timeframe-mode", choices=["strict_1m_on_5m", "5m_only"], default="strict_1m_on_5m")
    parser.add_argument("--signal-granularity", default="ONE_MINUTE", choices=sorted(GRANULARITY_TO_SECONDS))
    parser.add_argument("--context-granularity", default="FIVE_MINUTE", choices=sorted(GRANULARITY_TO_SECONDS))
    parser.add_argument("--lookback-bars", type=int, default=400)
    parser.add_argument("--poll-interval-seconds", type=float, default=5.0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--mode", choices=["paper", "live"], default="paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--credentials-path", default=None)
    parser.add_argument("--state-path", default="perps_breakout_state.json")
    parser.add_argument("--paper-cash", type=float, default=1000.0)
    parser.add_argument("--margin-type", choices=["CROSS", "ISOLATED"], default="CROSS")
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--reward-risk-ratio", type=float, default=1.5)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--min-box-atr-ratio", type=float, default=0.8)
    parser.add_argument("--min-volume-ratio", type=float, default=1.0)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--min-position-notional", type=float, default=10.0)
    parser.add_argument("--max-position-notional", type=float, default=500.0)
    parser.add_argument("--kalman-process-variance", type=float, default=0.0005)
    parser.add_argument("--kalman-measurement-variance", type=float, default=0.01)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.leverage <= 0 or args.leverage > 10:
        raise ValueError("Leverage must be greater than 0 and less than or equal to 10")
    if args.poll_interval_seconds <= 0:
        raise ValueError("Poll interval must be positive")
    if args.reward_risk_ratio <= 0:
        raise ValueError("Reward/risk ratio must be positive")
    if args.min_position_notional <= 0:
        raise ValueError("Minimum position notional must be positive")
    if args.max_position_notional < args.min_position_notional:
        raise ValueError("Maximum position notional must be greater than or equal to the minimum")


def build_config(args: argparse.Namespace) -> BreakoutConfig:
    config = BreakoutConfig(
        timeframe_mode=args.timeframe_mode,
        signal_granularity=args.signal_granularity,
        context_granularity=args.context_granularity,
        reward_risk_ratio=args.reward_risk_ratio,
        atr_period=args.atr_period,
        volume_window=args.volume_window,
        min_box_atr_ratio=args.min_box_atr_ratio,
        min_volume_ratio=args.min_volume_ratio,
        risk_fraction=args.risk_fraction,
        leverage=args.leverage,
        min_position_notional=args.min_position_notional,
        max_position_notional=args.max_position_notional,
        kalman_process_variance=args.kalman_process_variance,
        kalman_measurement_variance=args.kalman_measurement_variance,
    )
    return normalize_config(config)


def load_state(state_path: Path, product_id: str, paper_cash: float) -> dict[str, Any]:
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        schema_version = state.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"State file {state_path} uses schema {schema_version}; expected {SCHEMA_VERSION}. "
                "Start with a fresh state file for the breakout strategy."
            )
        existing_product_id = state.get("product_id")
        if existing_product_id and existing_product_id != product_id:
            raise ValueError(
                f"State file {state_path} belongs to {existing_product_id}, not {product_id}"
            )
        return state

    return {
        "schema_version": SCHEMA_VERSION,
        "product_id": product_id,
        "position": asdict(PositionState()),
        "paper_wallet": {
            "cash_balance": paper_cash,
            "equity": paper_cash,
            "realized_pnl": 0.0,
            "fees_paid": 0.0,
        },
        "last_signal": None,
        "last_execution": None,
        "last_processed_signal_time": None,
        "updated_at": None,
    }


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def position_from_state(state: dict[str, Any]) -> PositionState:
    return PositionState(**state.get("position", {}))


def parse_product_rules(product: dict[str, Any]) -> dict[str, float]:
    return {
        "base_increment": float(product.get("base_increment", "0.00000001")),
        "quote_increment": float(product.get("quote_increment", "0.01")),
        "base_min_size": float(product.get("base_min_size", "0.0")),
        "quote_min_size": float(product.get("quote_min_size", "0.0")),
    }


def resolve_fee_rate(client: CoinbaseAdvancedClient, args: argparse.Namespace) -> float:
    if args.taker_fee_rate is not None:
        return float(args.taker_fee_rate)
    fee_rates = client.get_fee_rates(product_type="FUTURE", product_venue="INTX")
    return float(fee_rates["taker_fee_rate"])


def drop_incomplete_candles(
    candles,
    granularity: str,
    now: datetime,
):
    granularity_seconds = GRANULARITY_TO_SECONDS[granularity]
    cutoff = int(now.timestamp()) - granularity_seconds
    latest_complete_start = pd_timestamp_from_seconds(cutoff).floor(f"{granularity_seconds}s")
    return candles[candles.index <= latest_complete_start].copy()


def pd_timestamp_from_seconds(epoch_seconds: int):
    import pandas as pd  # type: ignore

    return pd.Timestamp(epoch_seconds, unit="s", tz="UTC")


def fetch_completed_candles(
    client: CoinbaseAdvancedClient,
    *,
    product_id: str,
    granularity: str,
    lookback_bars: int,
    now: datetime,
):
    granularity_seconds = GRANULARITY_TO_SECONDS[granularity]
    end_time = int(now.timestamp())
    start_time = end_time - lookback_bars * granularity_seconds
    candles = client.fetch_candles(
        product_id=product_id,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity,
    )
    return drop_incomplete_candles(candles, granularity, now)


def resolve_available_equity(
    state: dict[str, Any],
    *,
    mode: str,
    live_balances: dict[str, float] | None = None,
) -> float:
    if mode == "paper":
        return float(state["paper_wallet"]["cash_balance"])
    balances = live_balances or {}
    return float(balances.get("USD", 0.0))


def update_paper_equity(
    state: dict[str, Any],
    position: PositionState,
    mark_price: float,
) -> None:
    wallet = state["paper_wallet"]
    wallet["equity"] = wallet["cash_balance"] + mark_to_market_pnl(position, mark_price)


def build_entry_plan(
    *,
    decision,
    position_side: str,
    entry_price: float,
    available_equity: float,
    config: BreakoutConfig,
    product_rules: dict[str, float],
    fee_rate: float,
) -> dict[str, Any]:
    size_plan = calculate_position_size(
        equity=available_equity,
        entry_price=entry_price,
        stop_reference_price=float(decision.stop_reference_price),
        direction=position_side,
        config=config,
        fee_rate=fee_rate,
    )
    base_size = round_to_increment(size_plan.quantity, product_rules["base_increment"])
    if base_size < product_rules["base_min_size"]:
        raise ValueError("Calculated base size falls below the exchange minimum")
    notional = base_size * entry_price
    if notional < max(config.min_position_notional, product_rules["quote_min_size"]):
        raise ValueError("Calculated notional falls below the exchange minimum")
    return {
        "base_size": base_size,
        "notional": notional,
        "stop_price": size_plan.stop_price,
        "take_profit_price": size_plan.take_profit_price,
        "margin_used": notional / config.leverage,
    }


def execute_entry(
    *,
    args: argparse.Namespace,
    client: CoinbaseAdvancedClient,
    state: dict[str, Any],
    product_id: str,
    decision,
    top_of_book: dict[str, float],
    product_rules: dict[str, float],
    config: BreakoutConfig,
    fee_rate: float,
) -> tuple[dict[str, Any], PositionState]:
    direction = str(decision.direction)
    side = "BUY" if direction == "LONG" else "SELL"
    entry_price = top_of_book["ask"] if direction == "LONG" else top_of_book["bid"]
    available_equity = resolve_available_equity(
        state,
        mode=args.mode,
        live_balances=client.get_available_balances() if args.mode == "live" else None,
    )
    entry_plan = build_entry_plan(
        decision=decision,
        position_side=direction,
        entry_price=entry_price,
        available_equity=available_equity,
        config=config,
        product_rules=product_rules,
        fee_rate=fee_rate,
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.mode == "paper":
        fee_paid = entry_plan["notional"] * fee_rate
        state["paper_wallet"]["cash_balance"] -= fee_paid
        state["paper_wallet"]["fees_paid"] += fee_paid
        position = PositionState(
            is_open=True,
            direction=direction,
            entry_price=entry_price,
            base_size=entry_plan["base_size"],
            notional=entry_plan["notional"],
            margin_used=entry_plan["margin_used"],
            leverage=config.leverage,
            stop_price=entry_plan["stop_price"],
            take_profit_price=entry_plan["take_profit_price"],
            breakout_high=decision.box.high,
            breakout_low=decision.box.low,
            signal_time=decision.signal_time,
            opened_at=timestamp,
            order_id="paper-entry",
        )
        return (
            {
                "action": "ENTER",
                "mode": "paper",
                "direction": direction,
                "fill_price": entry_price,
                "base_size": entry_plan["base_size"],
                "notional": entry_plan["notional"],
                "fee_paid": fee_paid,
                "stop_price": entry_plan["stop_price"],
                "take_profit_price": entry_plan["take_profit_price"],
            },
            position,
        )

    preview = client.preview_market_order(
        product_id=product_id,
        side=side,
        base_size=entry_plan["base_size"],
        leverage=args.leverage,
        margin_type=args.margin_type,
    )
    preview_max_leverage = float(preview.get("max_leverage", args.leverage) or args.leverage)
    if args.leverage > preview_max_leverage:
        raise ValueError(
            f"Requested leverage {args.leverage} exceeds preview max leverage {preview_max_leverage}"
        )
    if args.dry_run:
        return (
            {
                "action": "DRY_RUN_ENTRY",
                "mode": "live",
                "direction": direction,
                "base_size": entry_plan["base_size"],
                "reference_price": entry_price,
                "preview": preview,
            },
            PositionState(),
        )

    preview_id = preview.get("preview_id")
    order = client.create_market_order(
        product_id=product_id,
        side=side,
        base_size=entry_plan["base_size"],
        leverage=args.leverage,
        margin_type=args.margin_type,
        preview_id=preview_id,
    )
    order_id = (
        order.get("success_response", {}).get("order_id")
        or order.get("order_id")
        or "submitted"
    )
    position = PositionState(
        is_open=True,
        direction=direction,
        entry_price=entry_price,
        base_size=entry_plan["base_size"],
        notional=entry_plan["notional"],
        margin_used=entry_plan["margin_used"],
        leverage=config.leverage,
        stop_price=entry_plan["stop_price"],
        take_profit_price=entry_plan["take_profit_price"],
        breakout_high=decision.box.high,
        breakout_low=decision.box.low,
        signal_time=decision.signal_time,
        opened_at=timestamp,
        order_id=str(order_id),
    )
    return (
        {
            "action": "ENTER",
            "mode": "live",
            "direction": direction,
            "order_id": order_id,
            "base_size": entry_plan["base_size"],
            "reference_price": entry_price,
            "stop_price": entry_plan["stop_price"],
            "take_profit_price": entry_plan["take_profit_price"],
        },
        position,
    )


def execute_exit(
    *,
    args: argparse.Namespace,
    client: CoinbaseAdvancedClient,
    state: dict[str, Any],
    product_id: str,
    position: PositionState,
    exit_price: float,
    exit_reason: str,
    fee_rate: float,
) -> tuple[dict[str, Any], PositionState]:
    side = "SELL" if position.direction == "LONG" else "BUY"
    timestamp = datetime.now(timezone.utc).isoformat()

    if args.mode == "paper":
        notional = position.base_size * exit_price
        if position.direction == "LONG":
            realized_pnl = position.base_size * (exit_price - position.entry_price)
        else:
            realized_pnl = position.base_size * (position.entry_price - exit_price)
        fee_paid = notional * fee_rate
        state["paper_wallet"]["cash_balance"] += realized_pnl - fee_paid
        state["paper_wallet"]["realized_pnl"] += realized_pnl - fee_paid
        state["paper_wallet"]["fees_paid"] += fee_paid
        return (
            {
                "action": "EXIT",
                "mode": "paper",
                "direction": position.direction,
                "fill_price": exit_price,
                "reason": exit_reason,
                "realized_pnl": realized_pnl - fee_paid,
                "fee_paid": fee_paid,
                "closed_at": timestamp,
            },
            PositionState(),
        )

    preview = client.preview_market_order(
        product_id=product_id,
        side=side,
        base_size=position.base_size,
        margin_type=args.margin_type,
    )
    if args.dry_run:
        return (
            {
                "action": "DRY_RUN_EXIT",
                "mode": "live",
                "direction": position.direction,
                "reason": exit_reason,
                "reference_price": exit_price,
                "preview": preview,
            },
            position,
        )

    preview_id = preview.get("preview_id")
    order = client.create_market_order(
        product_id=product_id,
        side=side,
        base_size=position.base_size,
        margin_type=args.margin_type,
        preview_id=preview_id,
    )
    order_id = (
        order.get("success_response", {}).get("order_id")
        or order.get("order_id")
        or "submitted"
    )
    return (
        {
            "action": "EXIT",
            "mode": "live",
            "direction": position.direction,
            "reason": exit_reason,
            "reference_price": exit_price,
            "order_id": order_id,
            "closed_at": timestamp,
        },
        PositionState(),
    )


def build_output(
    *,
    timestamp: str,
    product_id: str,
    mode: str,
    decision: Any,
    execution: dict[str, Any] | None,
    position: PositionState,
    state: dict[str, Any],
    top_of_book: dict[str, float],
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "product_id": product_id,
        "mode": mode,
        "top_of_book": top_of_book,
        "decision": decision.to_dict() if decision is not None else None,
        "execution": execution,
        "position": asdict(position),
        "paper_wallet": state.get("paper_wallet") if mode == "paper" else None,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    state_path = Path(args.state_path)
    config = build_config(args)
    client = CoinbaseAdvancedClient(credentials_path=args.credentials_path)
    resolved_product_id = client.resolve_product_id(args.product_id)
    state = load_state(state_path, resolved_product_id, args.paper_cash)
    fee_rate = resolve_fee_rate(client, args)
    product = client.get_product(resolved_product_id)
    product_rules = parse_product_rules(product)
    position = position_from_state(state)
    history_bars = max(args.lookback_bars, required_history_bars(config) + 5)

    iteration = 0
    while True:
        now = datetime.now(timezone.utc)
        candles = fetch_completed_candles(
            client,
            product_id=resolved_product_id,
            granularity=config.signal_granularity,
            lookback_bars=history_bars,
            now=now,
        )
        if candles.empty:
            raise RuntimeError(f"No completed candles returned for {resolved_product_id}")

        best_bid_ask = client.get_best_bid_ask([resolved_product_id])
        top_of_book = client.get_top_of_book(best_bid_ask, resolved_product_id)
        execution: dict[str, Any] | None = None
        exit_happened = False

        if position.is_open:
            exit_decision = evaluate_live_exit(
                position,
                best_bid=top_of_book["bid"],
                best_ask=top_of_book["ask"],
            )
            if exit_decision.action == "EXIT":
                execution, position = execute_exit(
                    args=args,
                    client=client,
                    state=state,
                    product_id=resolved_product_id,
                    position=position,
                    exit_price=exit_decision.exit_price,
                    exit_reason=exit_decision.reason,
                    fee_rate=fee_rate,
                )
                exit_happened = execution["action"] == "EXIT"

        decision = evaluate_signal(candles=candles, config=config, position=position)
        latest_signal_time = decision.signal_time
        last_processed_signal_time = state.get("last_processed_signal_time")
        if exit_happened and latest_signal_time is not None:
            state["last_processed_signal_time"] = latest_signal_time
        if (
            not position.is_open
            and not exit_happened
            and latest_signal_time is not None
            and latest_signal_time != last_processed_signal_time
        ):
            state["last_processed_signal_time"] = latest_signal_time
            if decision.action in {"ENTER_LONG", "ENTER_SHORT"}:
                execution, position = execute_entry(
                    args=args,
                    client=client,
                    state=state,
                    product_id=resolved_product_id,
                    decision=decision,
                    top_of_book=top_of_book,
                    product_rules=product_rules,
                    config=config,
                    fee_rate=fee_rate,
                )
            else:
                execution = {
                    "action": "NO_ENTRY",
                    "mode": args.mode,
                    "reason": decision.reason,
                }

        state["position"] = asdict(position)
        if args.mode == "paper":
            update_paper_equity(
                state,
                position,
                mark_price=top_of_book["mid"] or float(candles["close"].iloc[-1]),
            )
        state["last_signal"] = decision.to_dict()
        if execution is not None:
            state["last_execution"] = execution
        state["updated_at"] = now.isoformat()
        save_state(state_path, state)

        output = build_output(
            timestamp=now.isoformat(),
            product_id=resolved_product_id,
            mode=args.mode,
            decision=decision,
            execution=execution,
            position=position,
            state=state,
            top_of_book=top_of_book,
        )
        print(json.dumps(output, indent=2))

        iteration += 1
        if args.max_iterations is not None and iteration >= args.max_iterations:
            break
        time.sleep(args.poll_interval_seconds)


if __name__ == "__main__":
    main()
