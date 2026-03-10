from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scalping_5min_momentum.back_testing.constants import (  # noqa: E402
    DEFAULT_GRANULARITY,
    DEFAULT_PERPETUAL_PRODUCTS,
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
from scalping_5min_momentum.scalping_strategy import ScalpingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the 5-minute momentum strategy across Coinbase perpetual futures.",
    )
    parser.add_argument("--products", nargs="+", default=DEFAULT_PERPETUAL_PRODUCTS)
    parser.add_argument("--granularity", default=DEFAULT_GRANULARITY)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--start-time", type=int, default=None)
    parser.add_argument("--end-time", type=int, default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--cache-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--credentials-path", default=None)
    parser.add_argument("--starting-cash", type=float, default=10000.0)
    parser.add_argument("--leverage", type=float, default=50.0)
    parser.add_argument("--position-allocation", type=float, default=0.81)
    parser.add_argument("--maker-fee-rate", type=float, default=None)
    parser.add_argument("--taker-fee-rate", type=float, default=None)
    parser.add_argument("--entry-liquidity", choices=["maker", "taker"], default="taker")
    parser.add_argument("--exit-liquidity", choices=["maker", "taker"], default="taker")
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--fast-ema", type=int, default=8)
    parser.add_argument("--slow-ema", type=int, default=21)
    parser.add_argument("--rsi-period", type=int, default=14)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--volume-window", type=int, default=20)
    parser.add_argument("--min-rsi-entry", type=float, default=55.0)
    parser.add_argument("--max-rsi-entry", type=float, default=78.0)
    parser.add_argument("--exit-rsi", type=float, default=48.0)
    parser.add_argument("--min-volume-ratio", type=float, default=0.8)
    parser.add_argument("--stop-atr", type=float, default=1.2)
    parser.add_argument("--take-profit-atr", type=float, default=1.8)
    parser.add_argument("--trailing-atr", type=float, default=1.0)
    parser.add_argument("--risk-fraction", type=float, default=0.02)
    parser.add_argument("--min-notional", type=float, default=10.0)
    parser.add_argument("--max-notional", type=float, default=100.0)
    return parser.parse_args()


def build_strategy_config(args: argparse.Namespace) -> ScalpingConfig:
    return ScalpingConfig(
        fast_ema=args.fast_ema,
        slow_ema=args.slow_ema,
        rsi_period=args.rsi_period,
        atr_period=args.atr_period,
        volume_window=args.volume_window,
        min_rsi_entry=args.min_rsi_entry,
        max_rsi_entry=args.max_rsi_entry,
        exit_rsi=args.exit_rsi,
        min_volume_ratio=args.min_volume_ratio,
        stop_atr_multiple=args.stop_atr,
        take_profit_atr_multiple=args.take_profit_atr,
        trailing_atr_multiple=args.trailing_atr,
        risk_fraction=args.risk_fraction,
        min_quote_notional=args.min_notional,
        max_quote_notional=args.max_notional,
    )


def build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        starting_cash=args.starting_cash,
        leverage=args.leverage,
        position_allocation=normalize_position_allocation(args.position_allocation),
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


def normalize_position_allocation(value: float) -> float:
    if value > 1:
        return value / 100.0
    return value


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


def main() -> None:
    args = parse_args()
    client = CoinbaseAdvancedClient(credentials_path=args.credentials_path)
    fee_rates = resolve_fee_rates(client, args)
    if args.maker_fee_rate is None:
        args.maker_fee_rate = float(fee_rates["maker_fee_rate"])
    if args.taker_fee_rate is None:
        args.taker_fee_rate = float(fee_rates["taker_fee_rate"])
    normalized_position_allocation = normalize_position_allocation(args.position_allocation)
    args.position_allocation = normalized_position_allocation
    strategy_config = build_strategy_config(args)
    backtest_config = build_backtest_config(args)
    start_time, end_time = resolve_time_range(args)

    package_root = Path(__file__).resolve().parent
    cache_dir = (package_root / args.cache_dir).resolve()
    output_dir = (package_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    if args.leverage > 50:
        warnings.append(
            "Coinbase help currently describes perpetual futures leverage up to 50x on "
            "eligible contracts. Backtests above 50x are hypothetical."
        )

    results: list[AssetBacktestResult] = []
    for product_id in args.products:
        request = HistoryRequest(
            product_id=product_id,
            granularity=args.granularity,
            start_time=start_time,
            end_time=end_time,
        )
        candles, cache_path = load_or_fetch_candles(
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
            granularity=args.granularity,
        )
        save_result_files(result=result, output_dir=output_dir)
        results.append(result)

    summary_rows = [result.summary() for result in results]
    portfolio_summary = build_portfolio_summary(results)
    summary_payload = {
        "portfolio_summary": portfolio_summary,
        "asset_summaries": summary_rows,
        "fee_rates": fee_rates,
        "execution_model": {
            "entry_liquidity": args.entry_liquidity,
            "exit_liquidity": args.exit_liquidity,
            "position_allocation": args.position_allocation,
            "leverage": args.leverage,
        },
        "time_range": {
            "start_time": start_time,
            "end_time": end_time,
        },
        "products": args.products,
        "granularity": args.granularity,
        "warnings": warnings,
    }

    summary_path = output_dir / "backtest_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    if summary_rows:
        import pandas as pd  # type: ignore

        pd.DataFrame(summary_rows).to_csv(output_dir / "asset_summaries.csv", index=False)
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
