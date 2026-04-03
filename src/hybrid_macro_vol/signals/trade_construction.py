"""
Trade construction engine.

Takes scored signals and proposes actual trade structures:
- Naked long puts/calls
- Vertical spreads (put spread, call spread)
- Layered convexity baskets
- Evaluates each structure's risk/reward under every scenario
"""

from __future__ import annotations

from datetime import date
from typing import Sequence

from src.hybrid_macro_vol.models import (
    OptionQuote,
    OptionSignal,
    OptionType,
    PricingResult,
    Scenario,
    ScenarioSet,
    TradeAction,
    TradeLeg,
    TradeRecommendation,
)
from src.hybrid_macro_vol.pricing.black_scholes import bs_price


def _scenario_value(
    S: float, K: float, T: float, r: float, sc: Scenario, otype: OptionType,
) -> float:
    shocked_spot = S * (1.0 + sc.spot_shock)
    shocked_rate = r + sc.rate_shift
    return bs_price(shocked_spot, K, T, shocked_rate, sc.volatility, otype)


# ─── Single-leg trades ───────────────────────────────

def long_put(
    signal: OptionSignal,
    scenario_set: ScenarioSet,
    risk_free_rate: float,
) -> TradeRecommendation:
    """Recommend a naked long put."""
    p = signal.pricing
    T = max(p.days_to_expiry / 365.0, 0.001)
    cost = p.market_mid

    scenario_pnl = {}
    for sc in scenario_set.scenarios:
        val = _scenario_value(p.underlying_price, p.strike, T, risk_free_rate, sc, OptionType.PUT)
        scenario_pnl[sc.name] = round((val - cost) * 100, 2)

    return TradeRecommendation(
        name=f"Long {p.strike:.0f} Put {p.expiry.isoformat()}",
        legs=[TradeLeg(
            action=TradeAction.BUY,
            option_type=OptionType.PUT,
            strike=p.strike,
            expiry=p.expiry,
            quantity=1,
            price=cost,
        )],
        max_loss=round(cost * 100, 2),
        max_gain=round((p.strike - cost) * 100, 2),  # at S=0
        breakeven=round(p.strike - cost, 2),
        net_debit=round(cost, 2),
        edge_score=signal.edge_score,
        scenario_pnl=scenario_pnl,
        rationale=f"Hybrid model values at {p.hybrid_value:.2f} vs market {p.market_mid:.2f} "
                  f"({p.edge_pct:+.1%} edge). {signal.notes}",
    )


def long_call(
    signal: OptionSignal,
    scenario_set: ScenarioSet,
    risk_free_rate: float,
) -> TradeRecommendation:
    """Recommend a naked long call."""
    p = signal.pricing
    T = max(p.days_to_expiry / 365.0, 0.001)
    cost = p.market_mid

    scenario_pnl = {}
    for sc in scenario_set.scenarios:
        val = _scenario_value(p.underlying_price, p.strike, T, risk_free_rate, sc, OptionType.CALL)
        scenario_pnl[sc.name] = round((val - cost) * 100, 2)

    return TradeRecommendation(
        name=f"Long {p.strike:.0f} Call {p.expiry.isoformat()}",
        legs=[TradeLeg(
            action=TradeAction.BUY,
            option_type=OptionType.CALL,
            strike=p.strike,
            expiry=p.expiry,
            quantity=1,
            price=cost,
        )],
        max_loss=round(cost * 100, 2),
        max_gain=None,  # unlimited for calls
        breakeven=round(p.strike + cost, 2),
        net_debit=round(cost, 2),
        edge_score=signal.edge_score,
        scenario_pnl=scenario_pnl,
        rationale=f"Hybrid model values at {p.hybrid_value:.2f} vs market {p.market_mid:.2f} "
                  f"({p.edge_pct:+.1%} edge). {signal.notes}",
    )


# ─── Spread trades ───────────────────────────────────

def put_spread(
    long_signal: OptionSignal,
    short_strike: float,
    chain_quotes: Sequence[OptionQuote],
    scenario_set: ScenarioSet,
    risk_free_rate: float,
) -> TradeRecommendation | None:
    """
    Construct a put spread: buy the long_signal put, sell a lower-strike put.

    This limits max loss but also caps convexity in the deep tail.
    Use when listed vol is expensive in the deep tail.
    """
    p = long_signal.pricing
    T = max(p.days_to_expiry / 365.0, 0.001)

    # Find the short leg quote
    short_quote = None
    for q in chain_quotes:
        if (q.option_type == OptionType.PUT
            and q.expiry == p.expiry
            and abs(q.strike - short_strike) < 0.01):
            short_quote = q
            break

    if short_quote is None:
        return None

    net_debit = p.market_mid - short_quote.mid
    if net_debit <= 0:
        return None

    width = p.strike - short_strike
    max_gain = width - net_debit
    max_loss = net_debit

    scenario_pnl = {}
    for sc in scenario_set.scenarios:
        long_val = _scenario_value(p.underlying_price, p.strike, T, risk_free_rate, sc, OptionType.PUT)
        short_val = _scenario_value(p.underlying_price, short_strike, T, risk_free_rate, sc, OptionType.PUT)
        spread_val = long_val - short_val
        scenario_pnl[sc.name] = round((spread_val - net_debit) * 100, 2)

    return TradeRecommendation(
        name=f"Put Spread {p.strike:.0f}/{short_strike:.0f} {p.expiry.isoformat()}",
        legs=[
            TradeLeg(
                action=TradeAction.BUY,
                option_type=OptionType.PUT,
                strike=p.strike,
                expiry=p.expiry,
                quantity=1,
                price=p.market_mid,
            ),
            TradeLeg(
                action=TradeAction.SELL,
                option_type=OptionType.PUT,
                strike=short_strike,
                expiry=p.expiry,
                quantity=1,
                price=short_quote.mid,
            ),
        ],
        max_loss=round(max_loss * 100, 2),
        max_gain=round(max_gain * 100, 2),
        breakeven=round(p.strike - net_debit, 2),
        net_debit=round(net_debit, 2),
        edge_score=long_signal.edge_score * 0.85,  # slightly less edge in a spread
        scenario_pnl=scenario_pnl,
        rationale=f"Put spread reduces cost to {net_debit:.2f} (vs {p.market_mid:.2f} naked). "
                  f"Max gain {max_gain:.2f} per contract if underlying below {short_strike:.0f}.",
    )


# ─── Convexity basket ────────────────────────────────

def convexity_basket(
    signals: Sequence[OptionSignal],
    scenario_set: ScenarioSet,
    risk_free_rate: float,
    max_legs: int = 4,
    total_budget: float = 5.0,
) -> TradeRecommendation:
    """
    Build a layered convexity basket: multiple puts across strikes/expiries
    to maximize tail payoff per dollar spent.

    Selects the options with the highest convexity_score within budget.
    """
    # Sort by convexity descending
    candidates = sorted(signals, key=lambda s: s.convexity_score, reverse=True)

    legs: list[TradeLeg] = []
    total_cost = 0.0
    selected_signals: list[OptionSignal] = []

    for sig in candidates:
        if len(legs) >= max_legs:
            break
        cost = sig.pricing.market_mid
        if total_cost + cost > total_budget:
            continue
        legs.append(TradeLeg(
            action=TradeAction.BUY,
            option_type=sig.pricing.option_type,
            strike=sig.pricing.strike,
            expiry=sig.pricing.expiry,
            quantity=1,
            price=cost,
        ))
        total_cost += cost
        selected_signals.append(sig)

    # Compute scenario P&L for the basket
    scenario_pnl: dict[str, float] = {}
    for sc in scenario_set.scenarios:
        total_pnl = 0.0
        for sig in selected_signals:
            p = sig.pricing
            T = max(p.days_to_expiry / 365.0, 0.001)
            val = _scenario_value(p.underlying_price, p.strike, T, risk_free_rate, sc, p.option_type)
            total_pnl += (val - p.market_mid) * 100
        scenario_pnl[sc.name] = round(total_pnl, 2)

    avg_edge = sum(s.edge_score for s in selected_signals) / len(selected_signals) if selected_signals else 0.0

    strikes_desc = "/".join(f"{s.pricing.strike:.0f}" for s in selected_signals)

    return TradeRecommendation(
        name=f"Convexity Basket {strikes_desc}",
        legs=legs,
        max_loss=round(total_cost * 100, 2),
        max_gain=None,
        net_debit=round(total_cost, 2),
        edge_score=avg_edge,
        scenario_pnl=scenario_pnl,
        rationale=f"Layered {len(legs)}-leg basket across strikes {strikes_desc}. "
                  f"Total cost: ${total_cost:.2f}/share. "
                  f"Maximizes tail convexity per dollar.",
    )


# ─── Recommendation engine ───────────────────────────

def recommend_trades(
    signals: list[OptionSignal],
    chain_quotes: Sequence[OptionQuote],
    scenario_set: ScenarioSet,
    risk_free_rate: float = 0.045,
    max_recommendations: int = 5,
) -> list[TradeRecommendation]:
    """
    Given ranked signals, produce the best trade recommendations.

    Generates single-leg and spread structures, ranks by edge and risk/reward.
    """
    recs: list[TradeRecommendation] = []

    # Top signals → naked longs
    for sig in signals[:3]:
        p = sig.pricing
        if p.option_type == OptionType.PUT:
            recs.append(long_put(sig, scenario_set, risk_free_rate))
        else:
            recs.append(long_call(sig, scenario_set, risk_free_rate))

    # Try put spreads for top put signals
    put_signals = [s for s in signals if s.pricing.option_type == OptionType.PUT]
    for sig in put_signals[:2]:
        p = sig.pricing
        # Try a spread with short leg 5-10% below
        for pct in [0.05, 0.10]:
            short_k = round(p.strike * (1 - pct))
            spread = put_spread(sig, short_k, chain_quotes, scenario_set, risk_free_rate)
            if spread:
                recs.append(spread)
                break

    # Convexity basket from top convexity signals
    high_convex = [s for s in signals if s.convexity_score > 0.3]
    if len(high_convex) >= 2:
        basket = convexity_basket(high_convex, scenario_set, risk_free_rate)
        if basket.legs:
            recs.append(basket)

    # Sort by edge_score
    recs.sort(key=lambda r: r.edge_score, reverse=True)
    return recs[:max_recommendations]
