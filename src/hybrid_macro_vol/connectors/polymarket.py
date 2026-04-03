"""Polymarket API connector for prediction market data."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from src.hybrid_macro_vol.models import PredictionMarketEvent

logger = logging.getLogger(__name__)

# Polymarket CLOB API base
BASE_URL = "https://clob.polymarket.com"
GAMMA_URL = "https://gamma-api.polymarket.com"

# Keywords to filter for macro-relevant markets
MACRO_KEYWORDS = [
    "recession", "gdp", "unemployment", "inflation", "cpi",
    "fed", "interest rate", "rate cut", "rate hike",
    "tariff", "trade war", "shutdown", "debt ceiling", "default",
    "s&p", "spy", "stock market", "crash", "bear market",
    "election", "president", "congress",
]


class PolymarketConnector:
    """Fetch prediction market data from Polymarket."""

    def __init__(self, timeout: float = 30.0):
        self._timeout = timeout

    def _get(self, url: str, params: dict | None = None) -> Any:
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(url, params=params or {})
            resp.raise_for_status()
            return resp.json()

    def get_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
    ) -> list[dict]:
        """Fetch markets from the Gamma API (human-readable metadata)."""
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
        }
        try:
            return self._get(f"{GAMMA_URL}/markets", params)
        except Exception as e:
            logger.error(f"Polymarket markets fetch failed: {e}")
            return []

    def search_macro_markets(
        self,
        keywords: list[str] | None = None,
        limit: int = 200,
    ) -> list[PredictionMarketEvent]:
        """Search for macro-relevant prediction markets."""
        kw = [k.lower() for k in (keywords or MACRO_KEYWORDS)]
        raw_markets = self.get_markets(limit=limit)
        events: list[PredictionMarketEvent] = []

        for m in raw_markets:
            title = (m.get("question") or m.get("title") or "").lower()
            if not any(k in title for k in kw):
                continue

            # Extract outcome prices — Polymarket stores outcomes as tokens
            outcomes = m.get("outcomes", [])
            outcome_prices = m.get("outcomePrices", [])

            # Find the "Yes" probability
            prob = 0.5
            if outcome_prices:
                try:
                    prices = [float(p) for p in outcome_prices]
                    # First outcome is typically "Yes"
                    prob = prices[0] if prices else 0.5
                except (ValueError, IndexError):
                    prob = 0.5

            # Parse settlement date
            settlement = None
            end_date = m.get("endDate") or m.get("end_date_iso")
            if end_date:
                try:
                    settlement = datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
                except (ValueError, TypeError):
                    pass

            event = PredictionMarketEvent(
                source="polymarket",
                event_id=str(m.get("id", m.get("condition_id", ""))),
                title=m.get("question") or m.get("title") or "",
                probability=prob,
                volume=_safe_float(m.get("volume") or m.get("volumeNum")),
                open_interest=_safe_float(m.get("liquidity") or m.get("liquidityNum")),
                settlement_date=settlement,
                category=m.get("category") or m.get("groupSlug"),
                timestamp=datetime.utcnow(),
                url=f"https://polymarket.com/event/{m.get('slug', m.get('condition_id', ''))}",
            )
            events.append(event)

        return events

    def get_market_by_id(self, condition_id: str) -> PredictionMarketEvent | None:
        """Fetch a specific market by condition ID."""
        try:
            data = self._get(f"{GAMMA_URL}/markets/{condition_id}")
            if not data:
                return None
            outcome_prices = data.get("outcomePrices", [])
            prob = float(outcome_prices[0]) if outcome_prices else 0.5
            return PredictionMarketEvent(
                source="polymarket",
                event_id=str(data.get("id", condition_id)),
                title=data.get("question", ""),
                probability=prob,
                volume=_safe_float(data.get("volumeNum")),
                settlement_date=None,
                category=data.get("category"),
                timestamp=datetime.utcnow(),
            )
        except Exception as e:
            logger.error(f"Polymarket market fetch failed for {condition_id}: {e}")
            return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
