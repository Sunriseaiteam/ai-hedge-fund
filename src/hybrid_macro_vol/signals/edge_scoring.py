"""
Edge scoring and signal classification.

Takes raw pricing results and produces scored, ranked signals
that identify the most mispriced options for a given macro event.
"""

from __future__ import annotations

from src.hybrid_macro_vol.models import (
    OptionSignal,
    PricingResult,
    SignalStrength,
)


def score_edge(result: PricingResult) -> float:
    """
    Compute a composite edge score for a pricing result.

    Combines:
    - Raw edge magnitude (hybrid_value - market_mid)
    - Edge as % of premium (cheaper options amplify signal)
    - Days to expiry penalty (theta drag on very short-dated)
    - Spread quality penalty (wide bid/ask = harder to capture)
    """
    if result.market_mid < 0.05:
        return 0.0

    # Raw edge contribution (normalized)
    raw_edge = result.edge
    pct_edge = result.edge_pct

    # Penalize very short-dated (< 14 DTE) — theta drag is brutal
    dte_factor = 1.0
    if result.days_to_expiry < 7:
        dte_factor = 0.3
    elif result.days_to_expiry < 14:
        dte_factor = 0.6
    elif result.days_to_expiry < 30:
        dte_factor = 0.85

    # Penalize wide bid/ask spreads
    spread = result.ask - result.bid
    spread_pct = spread / result.market_mid if result.market_mid > 0 else 1.0
    spread_factor = max(0.2, 1.0 - spread_pct * 2)  # 50% spread → 0.2 factor

    # Composite: weight raw edge and % edge
    score = (raw_edge * 0.4 + pct_edge * 10.0 * 0.6) * dte_factor * spread_factor

    return round(score, 4)


def score_liquidity(result: PricingResult) -> float:
    """
    Score liquidity on 0–1 scale based on bid/ask spread and volume proxy.
    """
    if result.market_mid < 0.01:
        return 0.0

    spread = result.ask - result.bid
    spread_pct = spread / result.market_mid

    # Tight spread = high score
    if spread_pct < 0.02:
        return 1.0
    elif spread_pct < 0.05:
        return 0.8
    elif spread_pct < 0.10:
        return 0.6
    elif spread_pct < 0.20:
        return 0.4
    elif spread_pct < 0.50:
        return 0.2
    return 0.1


def score_convexity(result: PricingResult) -> float:
    """
    Score convexity potential — how much upside does the option have
    in the worst-case scenario relative to its cost.

    High convexity = cheap option with big payoff in tail scenario.
    """
    if result.market_mid < 0.05:
        return 0.0

    # Find the max scenario value (i.e., worst-case-for-the-underlying payoff)
    if not result.scenario_values:
        return 0.0

    max_scenario_val = max(result.scenario_values.values())

    # Convexity = max_payoff / cost
    convexity_ratio = max_scenario_val / result.market_mid if result.market_mid > 0 else 0.0

    # Normalize to 0–1 range (ratio of 10x+ → 1.0)
    return min(1.0, convexity_ratio / 10.0)


def classify_signal(edge_score: float, edge_pct: float) -> SignalStrength:
    """Map edge metrics to a signal classification."""
    if edge_score > 2.0 and edge_pct > 0.20:
        return SignalStrength.STRONG_BUY
    elif edge_score > 1.0 and edge_pct > 0.10:
        return SignalStrength.BUY
    elif edge_score > 0.0 and edge_pct > 0.0:
        return SignalStrength.NEUTRAL
    elif edge_pct < -0.15:
        return SignalStrength.SELL
    else:
        return SignalStrength.AVOID


def generate_signal(result: PricingResult) -> OptionSignal:
    """Generate a complete signal from a pricing result."""
    edge = score_edge(result)
    liq = score_liquidity(result)
    conv = score_convexity(result)
    signal = classify_signal(edge, result.edge_pct)

    notes_parts = []
    if result.breakeven_probability is not None:
        notes_parts.append(f"Break-even event prob: {result.breakeven_probability:.1%}")
    if conv > 0.5:
        notes_parts.append("High convexity potential")
    if liq < 0.3:
        notes_parts.append("Low liquidity — caution")
    if result.edge_pct > 0.30:
        notes_parts.append(f"Edge: {result.edge_pct:.0%} of premium")

    return OptionSignal(
        pricing=result,
        signal=signal,
        edge_score=edge,
        liquidity_score=liq,
        convexity_score=conv,
        notes="; ".join(notes_parts),
    )
