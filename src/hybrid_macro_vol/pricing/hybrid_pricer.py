"""
Hybrid scenario-weighted option pricer.

Core idea:
  hybrid_value = Σ P(scenario_i) × BS_value(S_shocked_i, σ_i, r_i)

Compares this hybrid fair value against the listed market price to find edge.
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from src.hybrid_macro_vol.models import (
    OptionQuote,
    OptionType,
    PricingResult,
    Scenario,
    ScenarioSet,
)
from src.hybrid_macro_vol.pricing.black_scholes import bs_price


def price_option_under_scenario(
    S: float,
    K: float,
    T: float,
    r: float,
    scenario: Scenario,
    option_type: OptionType = OptionType.PUT,
) -> float:
    """Price a single option under one scenario."""
    shocked_spot = S * (1.0 + scenario.spot_shock)
    shocked_rate = r + scenario.rate_shift
    return bs_price(shocked_spot, K, T, shocked_rate, scenario.volatility, option_type)


def hybrid_price(
    S: float,
    K: float,
    T: float,
    r: float,
    scenarios: Sequence[Scenario],
    option_type: OptionType = OptionType.PUT,
) -> tuple[float, dict[str, float]]:
    """
    Compute scenario-weighted hybrid option price.

    Returns:
        (hybrid_value, {scenario_name: value_in_that_scenario})
    """
    total = 0.0
    breakdown: dict[str, float] = {}
    for sc in scenarios:
        val = price_option_under_scenario(S, K, T, r, sc, option_type)
        breakdown[sc.name] = val
        total += sc.probability * val
    return total, breakdown


def compute_breakeven_probability(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    scenario_set: ScenarioSet,
    option_type: OptionType = OptionType.PUT,
) -> float | None:
    """
    Find the event probability that makes hybrid_value == market_price.

    Uses bisection on the event probability (scenarios[1:] get scaled).
    Returns None if no solution found.
    """
    scenarios = scenario_set.scenarios
    if len(scenarios) < 2:
        return None

    no_event_val = price_option_under_scenario(S, K, T, r, scenarios[0], option_type)
    event_vals = []
    event_weights = []
    total_event_prob = sum(s.probability for s in scenarios[1:])

    for s in scenarios[1:]:
        event_vals.append(price_option_under_scenario(S, K, T, r, s, option_type))
        event_weights.append(s.probability / total_event_prob if total_event_prob > 0 else 1.0 / len(scenarios[1:]))

    def hybrid_at_prob(p_event: float) -> float:
        val = (1.0 - p_event) * no_event_val
        for w, v in zip(event_weights, event_vals):
            val += p_event * w * v
        return val

    # Bisect
    lo, hi = 0.0, 1.0
    f_lo = hybrid_at_prob(lo) - market_price
    f_hi = hybrid_at_prob(hi) - market_price

    if f_lo * f_hi > 0:
        # No root in [0,1] — return boundary
        return 0.0 if abs(f_lo) < abs(f_hi) else 1.0

    for _ in range(64):
        mid = (lo + hi) / 2
        f_mid = hybrid_at_prob(mid) - market_price
        if abs(f_mid) < 1e-8:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo = mid
            f_lo = f_mid

    return (lo + hi) / 2


def price_single_option(
    quote: OptionQuote,
    underlying_price: float,
    risk_free_rate: float,
    scenario_set: ScenarioSet,
    current_vol: float | None = None,
) -> PricingResult:
    """
    Full pricing pipeline for a single option quote.

    Computes BS value, hybrid value, edge, and breakeven probability.
    """
    T = max((quote.expiry - date.today()).days / 365.0, 0.001)
    vol = current_vol or quote.implied_vol or 0.20

    # Plain BS value at current market vol
    bs_val = bs_price(underlying_price, quote.strike, T, risk_free_rate, vol, quote.option_type)

    # Hybrid scenario-weighted value
    hybrid_val, scenario_breakdown = hybrid_price(
        underlying_price, quote.strike, T, risk_free_rate,
        scenario_set.scenarios, quote.option_type,
    )

    market_mid = quote.mid
    edge = hybrid_val - market_mid
    edge_pct = edge / market_mid if market_mid > 0.01 else 0.0

    # Break-even probability
    be_prob = compute_breakeven_probability(
        underlying_price, quote.strike, T, risk_free_rate,
        market_mid, scenario_set, quote.option_type,
    )

    return PricingResult(
        ticker=quote.ticker,
        expiry=quote.expiry,
        strike=quote.strike,
        option_type=quote.option_type,
        underlying_price=underlying_price,
        market_mid=market_mid,
        bs_value=bs_val,
        hybrid_value=hybrid_val,
        edge=edge,
        edge_pct=edge_pct,
        edge_per_contract=edge * 100,
        breakeven_probability=be_prob,
        scenario_values=scenario_breakdown,
        implied_vol=quote.implied_vol,
        days_to_expiry=max((quote.expiry - date.today()).days, 0),
        bid=quote.bid,
        ask=quote.ask,
    )


def scan_chain(
    chain: Sequence[OptionQuote],
    underlying_price: float,
    risk_free_rate: float,
    scenario_set: ScenarioSet,
    option_type: OptionType = OptionType.PUT,
    current_vol: float | None = None,
    min_oi: int = 0,
    min_mid: float = 0.05,
) -> list[PricingResult]:
    """
    Scan an entire option chain and return pricing results.

    Filters by option type, minimum open interest, and minimum mid price.
    """
    results = []
    for q in chain:
        if q.option_type != option_type:
            continue
        if q.mid < min_mid:
            continue
        if min_oi > 0 and (q.open_interest or 0) < min_oi:
            continue
        pr = price_single_option(q, underlying_price, risk_free_rate, scenario_set, current_vol)
        results.append(pr)

    results.sort(key=lambda r: r.edge, reverse=True)
    return results
