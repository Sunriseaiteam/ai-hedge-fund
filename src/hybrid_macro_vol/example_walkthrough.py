"""
Concrete Example: How a Polymarket Bet Flows Through the System
================================================================

Example: Polymarket shows "US Recession in 2026" at 42% probability.
The options market is pricing SPY puts at certain levels.
Does the options market agree with Polymarket? Where's the gap?

This script walks through every step with real numbers.
"""

from datetime import date, timedelta
from tabulate import tabulate

from src.hybrid_macro_vol.models import OptionQuote, OptionType, ScenarioSet, Scenario
from src.hybrid_macro_vol.pricing.black_scholes import bs_price
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
from src.hybrid_macro_vol.pricing.hybrid_pricer import hybrid_price, compute_breakeven_probability
from src.hybrid_macro_vol.signals.edge_scoring import generate_signal
from src.hybrid_macro_vol.pricing.hybrid_pricer import price_single_option


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  EXAMPLE: One Polymarket Bet → Option Mispricing Detection         ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP:
  You're looking at Polymarket and see:
    "Will the US enter a recession in 2026?"  →  YES: 42%

  Meanwhile, SPY is trading at $540.
  You're looking at a SPY 480 Put expiring in 90 days, listed at $3.50.

  QUESTION: Is that put cheap, fair, or expensive given 42% recession odds?

  Let's trace through the system step by step.
""")

    # ─── STEP 1: The Polymarket Signal ────────────────────────────
    polymarket_prob = 0.42  # 42% chance of recession

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: THE PREDICTION MARKET SIGNAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Polymarket says: P(recession) = {polymarket_prob:.0%}

  But this is just an event probability.
  It doesn't tell us HOW BAD the recession would be.
  That's where our scenario mapping comes in.
""")

    # ─── STEP 2: Map event probability into scenarios ─────────────
    maps = load_scenario_maps()
    base = maps["recession"]

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: MAP "42% RECESSION" INTO MARKET SCENARIOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  A "recession" isn't one thing. It could be mild or severe.
  Our scenario map breaks it down:
""")

    # Show base scenarios
    print("  DEFAULT scenario weights (before Polymarket):")
    for s in base.scenarios:
        print(f"    {s.name:20s}  P={s.probability:5.1%}  SPY move={s.spot_shock:+6.1%}  vol={s.volatility:5.0%}  │ {s.description}")

    # Now blend with Polymarket
    adjusted = blend_with_prediction_market(base, polymarket_prob)

    print(f"\n  ADJUSTED weights (using Polymarket's {polymarket_prob:.0%} recession probability):")
    print(f"  The 42% gets split across recession scenarios proportionally:\n")
    for s in adjusted.scenarios:
        print(f"    {s.name:20s}  P={s.probability:5.1%}  SPY move={s.spot_shock:+6.1%}  vol={s.volatility:5.0%}")

    event_prob_total = sum(s.probability for s in adjusted.scenarios[1:])
    print(f"\n  Total recession probability: {event_prob_total:.1%} (matches Polymarket)")
    print(f"  No-recession probability:   {adjusted.scenarios[0].probability:.1%}")

    # ─── STEP 3: Price the option under each scenario ─────────────
    S = 540.0   # SPY spot
    K = 480.0   # Strike
    T = 90/365  # 90 days
    r = 0.045   # Risk-free rate
    market_mid = 3.50  # Listed price

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: PRICE THE SPY 480 PUT UNDER EACH SCENARIO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Option: SPY 480 Put, 90 DTE
  Spot:   ${S:.0f}
  Listed: ${market_mid:.2f} (what the market charges)

  For each scenario, we shock the spot price and vol, then price with BS:
""")

    rows = []
    weighted_total = 0.0
    for s in adjusted.scenarios:
        shocked_S = S * (1 + s.spot_shock)
        shocked_r = r + s.rate_shift
        value = bs_price(shocked_S, K, T, shocked_r, s.volatility, OptionType.PUT)
        weighted_val = s.probability * value
        weighted_total += weighted_val

        rows.append({
            "Scenario": s.name,
            "Prob": f"{s.probability:.1%}",
            "SPY Price": f"${shocked_S:.0f}",
            "Vol": f"{s.volatility:.0%}",
            "Put Value": f"${value:.2f}",
            "Weighted": f"${weighted_val:.2f}",
        })

    print(tabulate(rows, headers="keys", tablefmt="simple"))

    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  Hybrid Fair Value = Σ (probability × put value)            │
  │                    = ${weighted_total:.2f}                            │
  │                                                             │
  │  Market Price       = ${market_mid:.2f}                              │
  │  Edge               = ${weighted_total - market_mid:.2f}  ({(weighted_total - market_mid)/market_mid:+.0%} of premium)      │
  └─────────────────────────────────────────────────────────────┘""")

    if weighted_total > market_mid:
        print(f"""
  INTERPRETATION: The put is CHEAP.
  If Polymarket is right about 42% recession odds, this put should
  cost ${weighted_total:.2f}, but the market only charges ${market_mid:.2f}.
  The options market is underpricing recession risk.
""")
    else:
        print(f"""
  INTERPRETATION: The put is EXPENSIVE or FAIR.
  The options market is already pricing enough (or more) downside risk
  than Polymarket implies.
""")

    # ─── STEP 4: What does the market think? ──────────────────────
    be_prob = compute_breakeven_probability(S, K, T, r, market_mid, adjusted, OptionType.PUT)

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 4: WHAT RECESSION PROBABILITY IS THE OPTIONS MARKET PRICING?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  We reverse-engineer the question:
  "What recession probability makes our model value = ${market_mid:.2f}?"

  Answer: {be_prob:.1%}

  Polymarket says:  {polymarket_prob:.0%} chance of recession
  Options imply:    {be_prob:.1%} chance of recession
  Gap:              {polymarket_prob - be_prob:+.1%}
""")

    if polymarket_prob > be_prob:
        print(f"  → Polymarket sees MORE risk than options are pricing.")
        print(f"    This is the signal: prediction markets and options disagree.")
    else:
        print(f"  → Options already price enough risk. No edge.")

    # ─── STEP 5: Compare across strikes ───────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 5: WHICH STRIKE HAS THE MOST EDGE?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  We scan multiple strikes to find where the gap is biggest:
""")

    strike_rows = []
    for K_test in [520, 510, 500, 490, 480, 470, 460, 450, 430, 400]:
        # Simulate a market price using BS with flat 20% vol (market's view)
        mkt_price = bs_price(S, K_test, T, r, 0.20, OptionType.PUT)
        if mkt_price < 0.05:
            continue

        hv, breakdown = hybrid_price(S, K_test, T, r, adjusted.scenarios, OptionType.PUT)
        edge = hv - mkt_price
        edge_pct = edge / mkt_price if mkt_price > 0 else 0

        # Convexity: max scenario payoff / cost
        max_payoff = max(breakdown.values())
        convexity = max_payoff / mkt_price if mkt_price > 0 else 0

        be = compute_breakeven_probability(S, K_test, T, r, mkt_price, adjusted, OptionType.PUT)

        strike_rows.append({
            "Strike": f"${K_test}",
            "Market": f"${mkt_price:.2f}",
            "Hybrid": f"${hv:.2f}",
            "Edge": f"${edge:.2f}",
            "Edge%": f"{edge_pct:+.0%}",
            "Mkt Implies": f"{be:.1%}" if be else "N/A",
            "Polymarket": f"{polymarket_prob:.0%}",
            "Convexity": f"{convexity:.1f}x",
            "Verdict": "CHEAP" if edge_pct > 0.15 else ("FAIR" if edge_pct > -0.05 else "RICH"),
        })

    print(tabulate(strike_rows, headers="keys", tablefmt="simple"))

    # ─── STEP 6: Trade recommendation ─────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6: TRADE RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Based on the analysis above, the system recommends:
""")

    # Find best edge
    best = max(strike_rows, key=lambda r: float(r["Edge"].replace("$", "")))
    print(f"  Best single-leg: Buy {best['Strike']} Put")
    print(f"    Cost:     {best['Market']}/share")
    print(f"    Model:    {best['Hybrid']}/share")
    print(f"    Edge:     {best['Edge']} ({best['Edge%']})")
    print(f"    Verdict:  {best['Verdict']}")

    print(f"""
  What this means in plain English:
  ─────────────────────────────────
  Polymarket crowds think there's a 42% chance of recession.
  But the options market is only pricing ~{be_prob:.0%} worth of downside.
  That {polymarket_prob - be_prob:+.0%} gap is the potential edge.

  If Polymarket is right, this put is underpriced.
  If the options market is right, you'll lose the premium.

  The system doesn't tell you WHO is right.
  It tells you WHERE they disagree and HOW to trade the gap.
""")

    # ─── STEP 7: Sensitivity ─────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 7: HOW DOES THE EDGE CHANGE WITH DIFFERENT PROBABILITIES?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  If you're not sure about 42%, here's how the 480 put looks at different probs:
""")

    K_focus = 480.0
    mkt_price_focus = bs_price(S, K_focus, T, r, 0.20, OptionType.PUT)

    sens_rows = []
    for p in [0.10, 0.20, 0.30, 0.42, 0.50, 0.60, 0.70]:
        adj = blend_with_prediction_market(base, p)
        hv, _ = hybrid_price(S, K_focus, T, r, adj.scenarios, OptionType.PUT)
        edge = hv - mkt_price_focus
        verdict = "CHEAP" if edge / mkt_price_focus > 0.15 else ("FAIR" if edge / mkt_price_focus > -0.05 else "RICH")
        marker = "  ← Polymarket" if p == 0.42 else ""

        sens_rows.append({
            "P(recession)": f"{p:.0%}",
            "Hybrid Value": f"${hv:.2f}",
            "Market Price": f"${mkt_price_focus:.2f}",
            "Edge": f"${edge:.2f}",
            "Verdict": verdict + marker,
        })

    print(tabulate(sens_rows, headers="keys", tablefmt="simple"))

    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Polymarket bet:     "US Recession 2026" = 42% YES
  2. Scenario mapping:   42% split into mild/severe/crisis proportionally
  3. Hybrid pricing:     Each scenario → shocked SPY + vol → BS price → weighted sum
  4. Market comparison:  Hybrid value vs listed price = edge
  5. Breakeven:          Options only pricing ~{be_prob:.0%} recession — gap of {polymarket_prob - be_prob:+.0%}
  6. Decision:           The gap suggests puts are cheap IF you trust Polymarket

  That's the entire pipeline. The dashboard shows all of this interactively.
""")


if __name__ == "__main__":
    main()
