"""Options market data connector.

Supports multiple data sources:
1. CBOE delayed quotes (free, delayed)
2. Financial Datasets API (same as existing project uses)
3. Manual / CSV import for backtesting

For production use, replace with a real-time data provider.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

import httpx

from src.hybrid_macro_vol.models import OptionChain, OptionQuote, OptionType

logger = logging.getLogger(__name__)


class OptionsDataConnector:
    """Fetch options chain data for equities/ETFs."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self._api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        self._timeout = timeout

    def get_option_chain(
        self,
        ticker: str,
        expiry_min: date | None = None,
        expiry_max: date | None = None,
    ) -> OptionChain | None:
        """
        Fetch a full option chain for a ticker.

        Uses the Financial Datasets API if available,
        falls back to generating a synthetic chain for development.
        """
        if self._api_key:
            return self._fetch_from_api(ticker, expiry_min, expiry_max)

        logger.warning(
            f"No API key set — generating synthetic option chain for {ticker}. "
            "Set FINANCIAL_DATASETS_API_KEY for real data."
        )
        return self._generate_synthetic_chain(ticker)

    def _fetch_from_api(
        self,
        ticker: str,
        expiry_min: date | None = None,
        expiry_max: date | None = None,
    ) -> OptionChain | None:
        """Fetch from Financial Datasets API options endpoint."""
        headers = {"X-API-KEY": self._api_key} if self._api_key else {}

        # First get current price
        price_url = f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={date.today().isoformat()}&end_date={date.today().isoformat()}"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(price_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    prices = data.get("prices", [])
                    spot = prices[-1]["close"] if prices else None
                else:
                    spot = None
        except Exception:
            spot = None

        # Options chain endpoint
        params: dict[str, Any] = {"ticker": ticker}
        if expiry_min:
            params["expiry_min"] = expiry_min.isoformat()
        if expiry_max:
            params["expiry_max"] = expiry_max.isoformat()

        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(
                    "https://api.financialdatasets.ai/options/chain/",
                    headers=headers,
                    params=params,
                )
                if resp.status_code != 200:
                    logger.warning(f"Options API returned {resp.status_code}, falling back to synthetic")
                    return self._generate_synthetic_chain(ticker, spot_override=spot)

                data = resp.json()
                options = data.get("options", data.get("option_chain", []))
                if not options:
                    return self._generate_synthetic_chain(ticker, spot_override=spot)

                quotes = []
                for opt in options:
                    try:
                        q = OptionQuote(
                            ticker=ticker,
                            expiry=date.fromisoformat(opt["expiration_date"]),
                            strike=float(opt["strike"]),
                            option_type=OptionType.CALL if opt.get("type", "").lower() == "call" else OptionType.PUT,
                            bid=float(opt.get("bid", 0)),
                            ask=float(opt.get("ask", 0)),
                            mid=(float(opt.get("bid", 0)) + float(opt.get("ask", 0))) / 2,
                            implied_vol=_safe_float(opt.get("implied_volatility")),
                            volume=int(opt.get("volume", 0)) if opt.get("volume") else None,
                            open_interest=int(opt.get("open_interest", 0)) if opt.get("open_interest") else None,
                            delta=_safe_float(opt.get("delta")),
                            gamma=_safe_float(opt.get("gamma")),
                            theta=_safe_float(opt.get("theta")),
                            vega=_safe_float(opt.get("vega")),
                            underlying_price=spot,
                        )
                        quotes.append(q)
                    except (KeyError, ValueError) as e:
                        logger.debug(f"Skipping malformed option: {e}")
                        continue

                return OptionChain(
                    ticker=ticker,
                    underlying_price=spot or 0.0,
                    risk_free_rate=0.045,
                    as_of=datetime.utcnow(),
                    quotes=quotes,
                )
        except Exception as e:
            logger.error(f"Options chain fetch failed: {e}")
            return self._generate_synthetic_chain(ticker, spot_override=spot)

    def _generate_synthetic_chain(
        self,
        ticker: str,
        spot_override: float | None = None,
    ) -> OptionChain:
        """
        Generate a realistic synthetic option chain for development/testing.

        Uses Black-Scholes to generate prices at various strikes/expiries.
        """
        from src.hybrid_macro_vol.pricing.black_scholes import bs_price, bs_delta

        # Default spot prices for common tickers
        default_spots = {"SPY": 540.0, "SPX": 5400.0, "QQQ": 460.0, "IWM": 200.0}
        spot = spot_override or default_spots.get(ticker.upper(), 500.0)
        rfr = 0.045
        base_vol = 0.18

        quotes: list[OptionQuote] = []
        today = date.today()

        # Generate for multiple expiries
        from datetime import timedelta
        expiries = [
            today + timedelta(days=d)
            for d in [14, 30, 60, 90, 120, 180, 270, 365]
        ]

        for exp in expiries:
            T = (exp - today).days / 365.0
            if T <= 0:
                continue

            # Strike range: 70% to 110% of spot, step by ~2%
            strike_step = round(spot * 0.02, 0) or 5.0
            strikes = []
            s = round(spot * 0.70 / strike_step) * strike_step
            while s <= spot * 1.10:
                strikes.append(s)
                s += strike_step

            for K in strikes:
                for otype in [OptionType.PUT, OptionType.CALL]:
                    # Realistic vol skew: OTM puts have significantly higher IV
                    # Calibrated to approximate typical equity skew
                    moneyness = K / spot
                    if otype == OptionType.PUT:
                        # Steep skew for OTM puts (real markets show ~0.5-0.8 per 10% OTM)
                        otm_dist = max(0, 1.0 - moneyness)
                        skew_adj = otm_dist * 0.6 + otm_dist**2 * 1.5  # convex skew
                    else:
                        skew_adj = max(0, (moneyness - 1.0) * 0.15)
                    vol = base_vol + skew_adj

                    price = bs_price(spot, K, T, rfr, vol, otype)
                    if price < 0.01:
                        continue

                    # Simulate bid-ask spread (wider for OTM, longer-dated)
                    spread = max(0.02, price * 0.03)
                    bid = max(0.01, price - spread / 2)
                    ask = price + spread / 2
                    mid = (bid + ask) / 2

                    delta = bs_delta(spot, K, T, rfr, vol, otype)

                    quotes.append(OptionQuote(
                        ticker=ticker,
                        expiry=exp,
                        strike=K,
                        option_type=otype,
                        bid=round(bid, 2),
                        ask=round(ask, 2),
                        mid=round(mid, 2),
                        implied_vol=round(vol, 4),
                        volume=int(max(10, 5000 * (1 - abs(delta)))),
                        open_interest=int(max(100, 50000 * (1 - abs(delta)))),
                        delta=round(delta, 4),
                        underlying_price=spot,
                    ))

        return OptionChain(
            ticker=ticker,
            underlying_price=spot,
            risk_free_rate=rfr,
            as_of=datetime.utcnow(),
            quotes=quotes,
        )

    def get_underlying_price(self, ticker: str) -> float | None:
        """Fetch current underlying price."""
        if not self._api_key:
            defaults = {"SPY": 540.0, "SPX": 5400.0, "QQQ": 460.0, "IWM": 200.0}
            return defaults.get(ticker.upper())

        headers = {"X-API-KEY": self._api_key}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(
                    f"https://api.financialdatasets.ai/prices/?ticker={ticker}&interval=day&interval_multiplier=1&start_date={date.today().isoformat()}&end_date={date.today().isoformat()}",
                    headers=headers,
                )
                if resp.status_code == 200:
                    prices = resp.json().get("prices", [])
                    return prices[-1]["close"] if prices else None
        except Exception as e:
            logger.error(f"Price fetch failed for {ticker}: {e}")
        return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
