"""
Signal ranking engine.

Scans options chains, scores every option against a scenario set,
and returns ranked lists of the most mispriced options — the ones
where prediction-market-implied probabilities diverge most from
what the options market is pricing.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd

from src.hybrid_macro_vol.models import (
    OptionChain,
    OptionQuote,
    OptionSignal,
    OptionType,
    PricingResult,
    ScenarioSet,
)
from src.hybrid_macro_vol.pricing.hybrid_pricer import scan_chain
from src.hybrid_macro_vol.signals.edge_scoring import generate_signal


def rank_opportunities(
    chain: OptionChain,
    scenario_set: ScenarioSet,
    option_type: OptionType = OptionType.PUT,
    min_oi: int = 50,
    min_mid: float = 0.10,
    min_dte: int = 7,
    max_dte: int = 400,
    top_n: int = 25,
) -> list[OptionSignal]:
    """
    Scan an option chain, price every contract under the hybrid model,
    score and rank by mispricing.

    Returns top_n signals sorted by edge_score descending.
    """
    today = date.today()

    # Filter quotes by DTE
    filtered_quotes = [
        q for q in chain.quotes
        if min_dte <= (q.expiry - today).days <= max_dte
    ]

    # Price all options under hybrid model
    results = scan_chain(
        chain=filtered_quotes,
        underlying_price=chain.underlying_price,
        risk_free_rate=chain.risk_free_rate,
        scenario_set=scenario_set,
        option_type=option_type,
        min_oi=min_oi,
        min_mid=min_mid,
    )

    # Generate signals
    signals = [generate_signal(r) for r in results]

    # Sort by edge_score, then convexity as tiebreaker
    signals.sort(key=lambda s: (s.edge_score, s.convexity_score), reverse=True)

    return signals[:top_n]


def rank_across_tickers(
    chains: dict[str, OptionChain],
    scenario_set: ScenarioSet,
    option_type: OptionType = OptionType.PUT,
    top_n: int = 25,
    **kwargs,
) -> list[OptionSignal]:
    """Rank opportunities across multiple tickers."""
    all_signals: list[OptionSignal] = []
    for ticker, chain in chains.items():
        sigs = rank_opportunities(
            chain=chain,
            scenario_set=scenario_set,
            option_type=option_type,
            top_n=top_n * 2,  # over-fetch, then trim
            **kwargs,
        )
        all_signals.extend(sigs)

    all_signals.sort(key=lambda s: (s.edge_score, s.convexity_score), reverse=True)
    return all_signals[:top_n]


def signals_to_dataframe(signals: list[OptionSignal]) -> pd.DataFrame:
    """Convert ranked signals to a clean DataFrame for display."""
    rows = []
    for sig in signals:
        p = sig.pricing
        rows.append({
            "ticker": p.ticker,
            "expiry": p.expiry.isoformat(),
            "strike": p.strike,
            "type": p.option_type.value,
            "DTE": p.days_to_expiry,
            "market_mid": round(p.market_mid, 2),
            "bs_value": round(p.bs_value, 2),
            "hybrid_value": round(p.hybrid_value, 2),
            "edge": round(p.edge, 2),
            "edge_pct": f"{p.edge_pct:.1%}",
            "edge_per_K": round(p.edge_per_contract, 0),
            "breakeven_prob": f"{p.breakeven_probability:.1%}" if p.breakeven_probability is not None else "N/A",
            "implied_vol": f"{p.implied_vol:.1%}" if p.implied_vol else "N/A",
            "signal": sig.signal.value,
            "edge_score": round(sig.edge_score, 2),
            "liquidity": round(sig.liquidity_score, 2),
            "convexity": round(sig.convexity_score, 2),
            "notes": sig.notes,
        })

    df = pd.DataFrame(rows)
    return df


def sensitivity_table(
    quote: OptionQuote,
    underlying_price: float,
    risk_free_rate: float,
    scenario_set: ScenarioSet,
    prob_range: Sequence[float] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60),
) -> pd.DataFrame:
    """
    Show how hybrid value changes across different event probabilities
    for a single option. Helps traders see where the edge appears/disappears.
    """
    from src.hybrid_macro_vol.pricing.hybrid_pricer import price_single_option
    from src.hybrid_macro_vol.pricing.scenarios import blend_with_prediction_market

    rows = []
    for prob in prob_range:
        adjusted = blend_with_prediction_market(scenario_set, prob)
        result = price_single_option(
            quote, underlying_price, risk_free_rate, adjusted,
        )
        rows.append({
            "event_prob": f"{prob:.0%}",
            "hybrid_value": round(result.hybrid_value, 2),
            "market_mid": round(result.market_mid, 2),
            "edge": round(result.edge, 2),
            "edge_pct": f"{result.edge_pct:.1%}",
        })

    return pd.DataFrame(rows)
