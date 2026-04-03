"""Kalshi API connector for prediction market data."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from src.hybrid_macro_vol.models import PredictionMarketEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

MACRO_KEYWORDS = [
    "recession", "gdp", "unemployment", "inflation", "cpi",
    "fed", "fomc", "interest rate", "rate cut",
    "tariff", "trade", "shutdown", "debt ceiling", "default",
    "s&p", "stock market", "crash",
]


class KalshiConnector:
    """
    Fetch prediction market data from Kalshi.

    Note: Kalshi requires authentication for most endpoints.
    Set api_key for authenticated access. Without it, only
    public/limited endpoints work.
    """

    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self._api_key = api_key
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{BASE_URL}{path}"
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, headers=self._headers(), params=params or {})
            resp.raise_for_status()
            return resp.json()

    def get_events(
        self,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 100,
    ) -> list[dict]:
        """Fetch events from Kalshi."""
        params: dict[str, Any] = {"limit": limit, "status": status}
        if series_ticker:
            params["series_ticker"] = series_ticker
        try:
            data = self._get("/events", params)
            return data.get("events", [])
        except Exception as e:
            logger.error(f"Kalshi events fetch failed: {e}")
            return []

    def get_markets(
        self,
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        status: str = "open",
        limit: int = 200,
    ) -> list[dict]:
        """Fetch markets (contracts) from Kalshi."""
        params: dict[str, Any] = {"limit": limit, "status": status}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        try:
            data = self._get("/markets", params)
            return data.get("markets", [])
        except Exception as e:
            logger.error(f"Kalshi markets fetch failed: {e}")
            return []

    def search_macro_markets(
        self,
        keywords: list[str] | None = None,
        limit: int = 200,
    ) -> list[PredictionMarketEvent]:
        """Search for macro-relevant prediction markets on Kalshi."""
        kw = [k.lower() for k in (keywords or MACRO_KEYWORDS)]
        raw = self.get_markets(limit=limit)
        events: list[PredictionMarketEvent] = []

        for m in raw:
            title = (m.get("title") or m.get("subtitle") or "").lower()
            if not any(k in title for k in kw):
                continue

            # Kalshi yes_price is in cents (0–100)
            yes_price = m.get("yes_bid") or m.get("last_price") or 50
            prob = yes_price / 100.0 if yes_price > 1 else yes_price

            settlement = None
            exp = m.get("expiration_time") or m.get("close_time")
            if exp:
                try:
                    settlement = datetime.fromisoformat(exp.replace("Z", "+00:00")).date()
                except (ValueError, TypeError):
                    pass

            event = PredictionMarketEvent(
                source="kalshi",
                event_id=m.get("ticker", ""),
                title=m.get("title") or m.get("subtitle") or "",
                probability=min(max(prob, 0.0), 1.0),
                bid=(m.get("yes_bid") or 0) / 100.0 if m.get("yes_bid") else None,
                ask=(m.get("yes_ask") or 0) / 100.0 if m.get("yes_ask") else None,
                volume=_safe_float(m.get("volume")),
                open_interest=_safe_float(m.get("open_interest")),
                settlement_date=settlement,
                category=m.get("category") or m.get("series_ticker"),
                timestamp=datetime.utcnow(),
            )
            events.append(event)

        return events

    def get_market_by_ticker(self, ticker: str) -> PredictionMarketEvent | None:
        """Fetch a specific market by ticker."""
        try:
            data = self._get(f"/markets/{ticker}")
            m = data.get("market", data)
            yes_price = m.get("yes_bid") or m.get("last_price") or 50
            prob = yes_price / 100.0 if yes_price > 1 else yes_price
            return PredictionMarketEvent(
                source="kalshi",
                event_id=m.get("ticker", ticker),
                title=m.get("title", ""),
                probability=min(max(prob, 0.0), 1.0),
                bid=(m.get("yes_bid") or 0) / 100.0 if m.get("yes_bid") else None,
                ask=(m.get("yes_ask") or 0) / 100.0 if m.get("yes_ask") else None,
                volume=_safe_float(m.get("volume")),
                open_interest=_safe_float(m.get("open_interest")),
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Kalshi market fetch failed for {ticker}: {e}")
            return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
