from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd  # type: ignore

from scalping_5min_momentum.coinbase_advanced import CoinbaseAdvancedClient


@dataclass(frozen=True)
class HistoryRequest:
    product_id: str
    granularity: str
    start_time: int
    end_time: int


def default_unix_range(days: int) -> tuple[int, int]:
    end_time = int(datetime.now(timezone.utc).timestamp())
    start_time = end_time - days * 24 * 60 * 60
    return start_time, end_time


def cache_file_path(cache_dir: Path, request: HistoryRequest) -> Path:
    sanitized_product = request.product_id.replace("-", "_")
    filename = (
        f"{sanitized_product}_{request.granularity}_{request.start_time}_{request.end_time}.csv"
    )
    return cache_dir / filename


def save_candles_to_cache(frame: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_path, index_label="timestamp")


def load_candles_from_cache(cache_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(cache_path, parse_dates=["timestamp"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.set_index("timestamp").sort_index()
    numeric_columns = ["open", "high", "low", "close", "volume"]
    frame[numeric_columns] = frame[numeric_columns].astype(float)
    return frame[numeric_columns]


def load_or_fetch_candles(
    client: CoinbaseAdvancedClient,
    request: HistoryRequest,
    cache_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, Path]:
    cache_path = cache_file_path(cache_dir, request)
    if cache_path.exists() and not refresh:
        return load_candles_from_cache(cache_path), cache_path

    frame = client.fetch_candles(
        product_id=request.product_id,
        start_time=request.start_time,
        end_time=request.end_time,
        granularity=request.granularity,
    )
    save_candles_to_cache(frame, cache_path)
    return frame, cache_path
