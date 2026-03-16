from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any, Optional

import jwt  # type: ignore
import pandas as pd  # type: ignore
import requests  # type: ignore
from cryptography.hazmat.primitives import serialization  # type: ignore
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, ed448  # type: ignore

API_HOST = "api.coinbase.com"
API_BASE_URL = f"https://{API_HOST}"
API_BASE_PATH = "/api/v3/brokerage"
MAX_CANDLES_PER_REQUEST = 350

GRANULARITY_TO_SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "THIRTY_MINUTE": 1800,
    "ONE_HOUR": 3600,
    "TWO_HOUR": 7200,
    "FOUR_HOUR": 14400,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}


@dataclass(frozen=True)
class CoinbaseCredentials:
    key_name: str
    private_key: str


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if text == "-0":
        return "0"
    return text


def round_to_increment(
    value: float | Decimal,
    increment: float | Decimal | str,
    rounding: str = ROUND_DOWN,
) -> float:
    increment_decimal = Decimal(str(increment))
    if increment_decimal <= 0:
        return float(value)
    value_decimal = Decimal(str(value))
    units = (value_decimal / increment_decimal).quantize(Decimal("1"), rounding=rounding)
    return float(units * increment_decimal)


def format_size(
    value: float | Decimal,
    increment: float | Decimal | str,
    rounding: str = ROUND_DOWN,
) -> str:
    rounded = round_to_increment(value, increment, rounding=rounding)
    return _format_decimal(Decimal(str(rounded)))


def _product_id_candidates(product_id: str) -> list[str]:
    candidates = [product_id]
    if product_id.endswith("-INTX"):
        candidates.append(product_id[: -len("-INTX")])
    elif product_id.endswith("-PERP"):
        candidates.append(f"{product_id}-INTX")
    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def load_coinbase_credentials(credentials_path: Optional[str] = None) -> CoinbaseCredentials:
    env_key_name = os.getenv("COINBASE_KEY_NAME")
    env_private_key = os.getenv("COINBASE_PRIVATE_KEY")
    if env_key_name and env_private_key:
        return CoinbaseCredentials(key_name=env_key_name, private_key=env_private_key)

    candidates = []
    if credentials_path:
        candidates.append(Path(credentials_path))
    else:
        module_dir = Path(__file__).resolve().parent
        quant_lab_dir = module_dir.parent
        candidates.extend(
            [
                Path("cdp_api_key.json"),
                Path("quant-lab/cdp_api_key.json"),
                quant_lab_dir / "cdp_api_key.json",
                Path(r"D:\Quant\quant-lab\cdp_api_key.json"),
                Path.home() / "cdp_api_key.json",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "name" not in payload or "privateKey" not in payload:
                raise ValueError(
                    f"Credentials file {candidate} must contain 'name' and 'privateKey' fields"
                )
            return CoinbaseCredentials(
                key_name=payload["name"],
                private_key=payload["privateKey"],
            )

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find Coinbase credentials. Searched: {searched}")


def _build_rest_jwt(method: str, path: str, credentials: CoinbaseCredentials) -> str:
    private_key = serialization.load_pem_private_key(
        credentials.private_key.encode("utf-8"),
        password=None,
    )

    if isinstance(private_key, ec.EllipticCurvePrivateKey):
        algorithm = "ES256"
    elif isinstance(private_key, (ed25519.Ed25519PrivateKey, ed448.Ed448PrivateKey)):
        algorithm = "EdDSA"
    else:
        raise TypeError(f"Unsupported private key type: {type(private_key)!r}")

    now = int(time.time())
    payload = {
        "sub": credentials.key_name,
        "iss": "cdp",
        "nbf": now,
        "exp": now + 120,
        "uri": f"{method.upper()} {API_HOST}{path}",
    }
    headers = {
        "kid": credentials.key_name,
        "nonce": secrets.token_hex(),
    }
    token = jwt.encode(payload, private_key, algorithm=algorithm, headers=headers)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


class CoinbaseAdvancedClient:
    def __init__(
        self,
        credentials: Optional[CoinbaseCredentials] = None,
        credentials_path: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: str = API_BASE_URL,
    ) -> None:
        self.credentials = credentials or load_coinbase_credentials(credentials_path)
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")
        self._resolved_product_ids: dict[str, str] = {}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
        timeout: int = 15,
    ) -> dict[str, Any]:
        token = _build_rest_jwt(method, path, self.credentials)
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        response = self.session.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            headers=headers,
            params=params,
            json=json_body,
            timeout=timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:1000]
            raise RuntimeError(
                f"Coinbase request failed for {method.upper()} {path} "
                f"with status {response.status_code}: {body}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Coinbase response for {method.upper()} {path} was not valid JSON"
            ) from exc

    def resolve_product_id(self, product_id: str) -> str:
        cached = self._resolved_product_ids.get(product_id)
        if cached:
            return cached
        for candidate in _product_id_candidates(product_id):
            try:
                self._request("GET", f"{API_BASE_PATH}/products/{candidate}")
            except RuntimeError:
                continue
            self._resolved_product_ids[product_id] = candidate
            self._resolved_product_ids[candidate] = candidate
            return candidate
        self._resolved_product_ids[product_id] = product_id
        return product_id

    def get_product(self, product_id: str) -> dict[str, Any]:
        resolved_product_id = self.resolve_product_id(product_id)
        return self._request("GET", f"{API_BASE_PATH}/products/{resolved_product_id}")

    def get_best_bid_ask(self, product_ids: list[str]) -> dict[str, Any]:
        resolved_product_ids = [self.resolve_product_id(product_id) for product_id in product_ids]
        return self._request(
            "GET",
            f"{API_BASE_PATH}/best_bid_ask",
            params={"product_ids": resolved_product_ids},
        )

    def list_accounts(self) -> list[dict[str, Any]]:
        accounts: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            params: dict[str, Any] = {"limit": 250}
            if cursor:
                params["cursor"] = cursor
            payload = self._request("GET", f"{API_BASE_PATH}/accounts", params=params)
            accounts.extend(payload.get("accounts", []))
            if not payload.get("has_next"):
                return accounts
            cursor = payload.get("cursor")
            if not cursor:
                return accounts

    def get_available_balances(self) -> dict[str, float]:
        balances: dict[str, float] = {}
        for account in self.list_accounts():
            currency = account.get("currency")
            available = account.get("available_balance", {}).get("value")
            if currency is None or available is None:
                continue
            balances[currency] = float(available)
        return balances

    def get_transaction_summary(
        self,
        *,
        product_type: Optional[str] = None,
        contract_expiry_type: Optional[str] = None,
        product_venue: Optional[str] = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if product_type:
            params["product_type"] = product_type
        if contract_expiry_type:
            params["contract_expiry_type"] = contract_expiry_type
        if product_venue:
            params["product_venue"] = product_venue
        return self._request("GET", f"{API_BASE_PATH}/transaction_summary", params=params)

    def get_fee_rates(
        self,
        *,
        product_type: str = "FUTURE",
        product_venue: str = "INTX",
    ) -> dict[str, Any]:
        payload = self.get_transaction_summary(
            product_type=product_type,
            product_venue=product_venue,
        )
        fee_tier = payload.get("fee_tier", {})
        return {
            "pricing_tier": fee_tier.get("pricing_tier"),
            "maker_fee_rate": float(fee_tier.get("maker_fee_rate", 0.0) or 0.0),
            "taker_fee_rate": float(fee_tier.get("taker_fee_rate", 0.0) or 0.0),
            "margin_rate": float(payload.get("margin_rate", 0.0) or 0.0),
        }

    def get_candles(
        self,
        product_id: str,
        start_time: int,
        end_time: int,
        granularity: str,
        limit: int = MAX_CANDLES_PER_REQUEST,
    ) -> pd.DataFrame:
        resolved_product_id = self.resolve_product_id(product_id)
        path = f"{API_BASE_PATH}/products/{resolved_product_id}/candles"
        payload = self._request(
            "GET",
            path,
            params={
                "start": str(start_time),
                "end": str(end_time),
                "granularity": granularity,
                "limit": limit,
            },
        )
        candles = payload.get("candles", [])
        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        frame = pd.DataFrame(candles)
        required_columns = ["start", "low", "high", "open", "close", "volume"]
        missing = [column for column in required_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Coinbase candles response missing columns: {missing}")

        frame["start"] = pd.to_datetime(pd.to_numeric(frame["start"]), unit="s", utc=True)
        frame = frame.sort_values("start").set_index("start")
        numeric_columns = ["open", "high", "low", "close", "volume"]
        frame[numeric_columns] = frame[numeric_columns].astype(float)
        return frame[numeric_columns]

    def fetch_candles(
        self,
        product_id: str,
        start_time: int,
        end_time: int,
        granularity: str,
    ) -> pd.DataFrame:
        granularity_seconds = GRANULARITY_TO_SECONDS[granularity]
        chunk_span = granularity_seconds * MAX_CANDLES_PER_REQUEST
        frames: list[pd.DataFrame] = []
        cursor = start_time

        while cursor < end_time:
            chunk_end = min(end_time, cursor + chunk_span)
            frame = self.get_candles(
                product_id=product_id,
                start_time=cursor,
                end_time=chunk_end,
                granularity=granularity,
            )
            if not frame.empty:
                frames.append(frame)
            cursor = chunk_end

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    def preview_market_order(
        self,
        product_id: str,
        side: str,
        *,
        quote_size: Optional[float] = None,
        base_size: Optional[float] = None,
        leverage: Optional[float] = None,
        margin_type: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = self._build_market_order_payload(
            product_id=self.resolve_product_id(product_id),
            side=side,
            quote_size=quote_size,
            base_size=base_size,
            leverage=leverage,
            margin_type=margin_type,
        )
        return self._request("POST", f"{API_BASE_PATH}/orders/preview", json_body=payload)

    def create_market_order(
        self,
        product_id: str,
        side: str,
        *,
        quote_size: Optional[float] = None,
        base_size: Optional[float] = None,
        leverage: Optional[float] = None,
        margin_type: Optional[str] = None,
        preview_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = self._build_market_order_payload(
            product_id=self.resolve_product_id(product_id),
            side=side,
            quote_size=quote_size,
            base_size=base_size,
            leverage=leverage,
            margin_type=margin_type,
            client_order_id=client_order_id,
        )
        if preview_id:
            payload["preview_id"] = preview_id
        return self._request("POST", f"{API_BASE_PATH}/orders", json_body=payload)

    def _build_market_order_payload(
        self,
        *,
        product_id: str,
        side: str,
        quote_size: Optional[float] = None,
        base_size: Optional[float] = None,
        leverage: Optional[float] = None,
        margin_type: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if quote_size is None and base_size is None:
            raise ValueError("Either quote_size or base_size must be supplied")

        configuration: dict[str, Any] = {}
        if quote_size is not None:
            configuration["quote_size"] = _format_decimal(Decimal(str(quote_size)))
        if base_size is not None:
            configuration["base_size"] = _format_decimal(Decimal(str(base_size)))
        if leverage is not None:
            configuration["leverage"] = _format_decimal(Decimal(str(leverage)))
        if margin_type is not None:
            configuration["margin_type"] = margin_type.upper()

        return {
            "client_order_id": client_order_id or str(uuid.uuid4()),
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "market_market_ioc": configuration,
            },
        }

    @staticmethod
    def get_top_of_book(best_bid_ask_payload: dict[str, Any], product_id: str) -> dict[str, float]:
        product_candidates = set(_product_id_candidates(product_id))
        for pricebook in best_bid_ask_payload.get("pricebooks", []):
            if pricebook.get("product_id") not in product_candidates:
                continue
            bid = float((pricebook.get("bids") or [{}])[0].get("price", 0.0))
            ask = float((pricebook.get("asks") or [{}])[0].get("price", 0.0))
            return {
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2 if bid and ask else 0.0,
            }
        raise ValueError(f"No pricebook returned for product {product_id}")


__all__ = [
    "API_BASE_PATH",
    "CoinbaseAdvancedClient",
    "CoinbaseCredentials",
    "GRANULARITY_TO_SECONDS",
    "format_size",
    "load_coinbase_credentials",
    "round_to_increment",
]
