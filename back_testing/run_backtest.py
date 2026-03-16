from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scalping_5min_momentum.back_testing.constants import (  # noqa: E402
    DEFAULT_CONTEXT_GRANULARITY,
    DEFAULT_PERPETUAL_PRODUCTS,
    DEFAULT_SIGNAL_GRANULARITY,
)
from scalping_5min_momentum.back_testing.data_sources import (  # noqa: E402
    HistoryRequest,
    default_unix_range,
    load_or_fetch_candles,
)
from scalping_5min_momentum.back_testing.engine import (  # noqa: E402
    AssetBacktestResult,
    BacktestConfig,
    run_backtest_for_asset,
)
from scalping_5min_momentum.coinbase_advanced import CoinbaseAdvancedClient  # noqa: E402
from scalping_5min_momentum.scalping_strategy import (  # noqa: E402
    BreakoutConfig,
    TIMEFRAME_MODE_FIVE_ONLY,
    TIMEFRAME_MODE_STRICT,
    normalize_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the perps box-breakout scalping strategy across Coinbase perpetual futures.",
    )
    parser.add_argument("--products", nargs="+", default=DEFAULT_PERPETUAL_PRODUCTS)
    parser.add_argument(
        "--timeframe-mode",
        choices=[TIMEFRAME_MODE_STRICT, TIMEFRAME_MODE_FIVE_ONLY],
        default=TIMEFRAME_MODE_STRICT,
    )
    parser.add_argument("--signal-granularity", default=DEFAULT_SIGNAL_GRANULARITY)
    parser.add_argument("--context-granularity", default=DEFAULT_CONTEXT_GRANULARITY)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-time", type=int, default=None)
    parser.add_argument("--end-time", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--credentials-path", default=None)
    parser.add_argument("--starting-cash", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=2.0)
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--entry-liquidity", choices=["maker", "taker"], default="taker")
    parser.add_argument("--exit-liquidity", choices=["maker", "taker"], default="taker")
    parser.add_argument("--slippage-bps", type=float, default=2.0)
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


def build_strategy_config(args: argparse.Namespace) -> BreakoutConfig:
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


def build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        starting_cash=args.starting_cash,
        leverage=args.leverage,
        maker_fee_rate=float(args.maker_fee_rate or 0.0),
        taker_fee_rate=float(args.taker_fee_rate or 0.0),
        entry_liquidity=args.entry_liquidity,
        exit_liquidity=args.exit_liquidity,
        slippage_bps=args.slippage_bps,
    )


def resolve_time_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.start_time is not None and args.end_time is not None:
        return args.start_time, args.end_time
    return default_unix_range(args.days)


def save_result_files(
    result: AssetBacktestResult,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    product_slug = result.product_id.replace("-", "_")
    result.trades_frame.to_csv(output_dir / f"{product_slug}_trades.csv", index=False)
    result.equity_curve.to_csv(output_dir / f"{product_slug}_equity.csv", index_label="timestamp")


def build_portfolio_summary(results: list[AssetBacktestResult]) -> dict[str, float | int]:
    total_starting_cash = sum(result.starting_cash for result in results)
    total_final_equity = sum(result.final_equity for result in results)
    total_trades = sum(result.trades_count for result in results)
    total_fees_paid = sum(result.fees_paid for result in results)
    portfolio_return_pct = (
        ((total_final_equity / total_starting_cash) - 1) * 100 if total_starting_cash else 0.0
    )
    average_win_rate = (
        sum(result.win_rate_pct for result in results) / len(results) if results else 0.0
    )
    average_drawdown = (
        sum(result.max_drawdown_pct for result in results) / len(results) if results else 0.0
    )
    return {
        "assets_tested": len(results),
        "total_starting_cash": round(total_starting_cash, 2),
        "total_final_equity": round(total_final_equity, 2),
        "portfolio_return_pct": round(portfolio_return_pct, 4),
        "total_trades": total_trades,
        "total_fees_paid": round(total_fees_paid, 4),
        "average_win_rate_pct": round(average_win_rate, 2),
        "average_max_drawdown_pct": round(average_drawdown, 4),
    }


def resolve_fee_rates(
    client: CoinbaseAdvancedClient,
    args: argparse.Namespace,
) -> dict[str, float | str | None]:
    if args.maker_fee_rate is not None and args.taker_fee_rate is not None:
        return {
            "pricing_tier": "manual_override",
            "maker_fee_rate": float(args.maker_fee_rate),
            "taker_fee_rate": float(args.taker_fee_rate),
        }
    return client.get_fee_rates(product_type="FUTURE", product_venue="INTX")


def validate_args(args: argparse.Namespace) -> None:
    if args.leverage <= 0 or args.leverage > 10:
        raise ValueError("Leverage must be greater than 0 and less than or equal to 10")
    if args.min_position_notional <= 0:
        raise ValueError("Minimum position notional must be positive")
    if args.max_position_notional < args.min_position_notional:
        raise ValueError("Maximum position notional must be greater than or equal to the minimum")
    if args.reward_risk_ratio <= 0:
        raise ValueError("Reward/risk ratio must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    client = CoinbaseAdvancedClient(credentials_path=args.credentials_path)
    fee_rates = resolve_fee_rates(client, args)
    if args.maker_fee_rate is None:
        args.maker_fee_rate = float(fee_rates["maker_fee_rate"])
    if args.taker_fee_rate is None:
        args.taker_fee_rate = float(fee_rates["taker_fee_rate"])

    strategy_config = build_strategy_config(args)
    backtest_config = build_backtest_config(args)
    start_time, end_time = resolve_time_range(args)

    package_root = Path(__file__).resolve().parent
    cache_dir = (package_root / args.cache_dir).resolve()
    output_dir = (package_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[AssetBacktestResult] = []
    for product_id in args.products:
        request = HistoryRequest(
            product_id=product_id,
            granularity=strategy_config.signal_granularity,
            start_time=start_time,
            end_time=end_time,
        )
        candles, _cache_path = load_or_fetch_candles(
            client=client,
            request=request,
            cache_dir=cache_dir,
            refresh=args.refresh_cache,
        )
        if candles.empty:
            continue
        result = run_backtest_for_asset(
            product_id=product_id,
            candles=candles,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            granularity=strategy_config.signal_granularity,
        )
        save_result_files(result=result, output_dir=output_dir)
        results.append(result)

    summary_rows = [result.summary() for result in results]
    portfolio_summary = build_portfolio_summary(results)
    summary_payload = {
        "portfolio_summary": portfolio_summary,
        "asset_summaries": summary_rows,
        "fee_rates": fee_rates,
        "strategy": {
            "timeframe_mode": strategy_config.timeframe_mode,
            "signal_granularity": strategy_config.signal_granularity,
            "context_granularity": strategy_config.context_granularity,
            "reward_risk_ratio": strategy_config.reward_risk_ratio,
            "atr_period": strategy_config.atr_period,
            "volume_window": strategy_config.volume_window,
            "min_box_atr_ratio": strategy_config.min_box_atr_ratio,
            "min_volume_ratio": strategy_config.min_volume_ratio,
            "risk_fraction": strategy_config.risk_fraction,
            "leverage": strategy_config.leverage,
            "kalman_process_variance": strategy_config.kalman_process_variance,
            "kalman_measurement_variance": strategy_config.kalman_measurement_variance,
        },
        "execution_model": {
            "entry_liquidity": args.entry_liquidity,
            "exit_liquidity": args.exit_liquidity,
            "slippage_bps": args.slippage_bps,
            "signal_on_close_fill_next_open": True,
        },
        "time_range": {
            "start_time": start_time,
            "end_time": end_time,
        },
        "products": args.products,
    }

    summary_path = output_dir / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if summary_rows:
        import pandas as pd  # type: ignore

        pd.DataFrame(summary_rows).to_csv(output_dir / "asset_summaries.csv", index=False)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
