"""
Skew Difference Visualizer
===========================

Shows exactly how the options market's implied volatility skew differs from
what the hybrid scenario model implies — and where the gaps create edge.

The key insight: options markets price risk through the vol surface (skew).
Our model prices risk through discrete crash scenarios. When these two
approaches disagree, that's where mispriced options live.
"""

from datetime import date, timedelta
from tabulate import tabulate

from src.hybrid_macro_vol.models import OptionType
from src.hybrid_macro_vol.pricing.black_scholes import bs_price, implied_vol
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
from src.hybrid_macro_vol.pricing.hybrid_pricer import hybrid_price


def main():
    S = 540.0
    T = 90 / 365.0
    r = 0.045

    maps = load_scenario_maps()
    base = maps["recession"]
    adjusted = blend_with_prediction_market(base, 0.42)  # Polymarket: 42% recession

    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  SKEW DIFFERENCE: Market Vol Skew vs Hybrid-Model-Implied Skew        ║
╚══════════════════════════════════════════════════════════════════════════╝

  SPY = $540  |  90 DTE  |  Polymarket recession prob = 42%

  Two ways to price options:
    1. MARKET SKEW  — each strike has its own implied vol (the smile/skew)
    2. HYBRID MODEL — discrete crash scenarios, each with a flat vol

  We compare them strike-by-strike to find where they disagree.
""")

    # ─── Build the comparison ─────────────────────────────────────
    base_vol = 0.18  # ATM vol

    strikes = [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540]

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("PART 1: Market Skew vs Hybrid-Implied Skew")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    rows = []
    for K in strikes:
        moneyness = K / S
        otm_pct = (1.0 - moneyness) * 100

        # Market skew: simulate realistic equity skew
        # OTM puts have higher IV, roughly linear + convex tail
        otm_dist = max(0, 1.0 - moneyness)
        market_vol = base_vol + otm_dist * 0.6 + otm_dist**2 * 1.5
        market_price = bs_price(S, K, T, r, market_vol, OptionType.PUT)

        # Hybrid model price (scenario-weighted)
        hybrid_val, breakdown = hybrid_price(S, K, T, r, adjusted.scenarios, OptionType.PUT)

        # Reverse-engineer: what IV does the hybrid price imply?
        hybrid_implied = implied_vol(hybrid_val, S, K, T, r, OptionType.PUT)

        # The difference
        if hybrid_implied is not None:
            vol_diff = hybrid_implied - market_vol
            vol_diff_str = f"{vol_diff:+.1%}"
        else:
            vol_diff = 0
            vol_diff_str = "N/A"

        edge = hybrid_val - market_price
        edge_pct = edge / market_price * 100 if market_price > 0.01 else 0

        # Visual bar for the vol difference
        if hybrid_implied is not None:
            bar_len = int(min(abs(vol_diff) * 200, 30))
            if vol_diff > 0.005:
                bar = "█" * bar_len + " CHEAP"
            elif vol_diff < -0.005:
                bar = "▒" * bar_len + " RICH"
            else:
                bar = "═ FAIR"
        else:
            bar = ""

        rows.append({
            "Strike": f"${K}",
            "OTM%": f"{otm_pct:.0f}%",
            "Mkt IV": f"{market_vol:.1%}",
            "Hybrid IV": f"{hybrid_implied:.1%}" if hybrid_implied else "N/A",
            "IV Gap": vol_diff_str,
            "Mkt $": f"{market_price:.2f}",
            "Hybrid $": f"{hybrid_val:.2f}",
            "Edge $": f"{edge:+.2f}",
            "": bar,
        })

    print(tabulate(rows, headers="keys", tablefmt="simple"))

    # ─── Part 2: Decompose WHY the skews differ ──────────────────
    print(f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2: WHY DO THEY DIFFER?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The market prices each strike with a CONTINUOUS vol surface.
  Our model prices with DISCRETE crash scenarios.

  Here's what each scenario contributes to the hybrid price:
""")

    detail_strikes = [520, 500, 480, 460, 440, 420, 400]
    for K in detail_strikes:
        hybrid_val, breakdown = hybrid_price(S, K, T, r, adjusted.scenarios, OptionType.PUT)

        otm_dist = max(0, 1.0 - K / S)
        market_vol = base_vol + otm_dist * 0.6 + otm_dist**2 * 1.5
        market_price = bs_price(S, K, T, r, market_vol, OptionType.PUT)

        print(f"  ── ${K} Put ({'ATM' if K >= S else f'{(1-K/S)*100:.0f}% OTM'}) ──")
        print(f"  Market price: ${market_price:.2f} (at {market_vol:.1%} IV)")
        print(f"  Hybrid price: ${hybrid_val:.2f}")
        print(f"  Breakdown:")

        for s in adjusted.scenarios:
            val = breakdown[s.name]
            weighted = s.probability * val
            pct_of_total = weighted / hybrid_val * 100 if hybrid_val > 0 else 0
            bar_len = int(pct_of_total / 3)
            bar = "█" * bar_len

            shocked_spot = S * (1 + s.spot_shock)
            itm = "ITM" if shocked_spot < K else "OTM"

            print(f"    {s.name:20s}  P={s.probability:5.1%} × ${val:8.2f} = ${weighted:6.2f}  "
                  f"({pct_of_total:4.1f}% of value)  {bar}  [SPY→${shocked_spot:.0f} {itm}]")

        print()

    # ─── Part 3: The skew crossover point ─────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3: WHERE DOES THE HYBRID MODEL DISAGREE MOST?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The hybrid model's implied vol is ABOVE market skew for most OTM puts.
  This means the market's continuous skew doesn't fully capture the
  probability of discrete crash events that Polymarket is pricing in.

  Key insight:
  ─────────────────────────────────────────────────────────────────
  The market skew is a smooth curve that rises as you go OTM.
  But crashes aren't smooth — they're jumps.

  At 42% recession probability:
  - A mild recession (SPY→$475) puts the 480 strike IN THE MONEY
  - A severe recession (SPY→$389) makes the 480 put worth $91
  - A crisis (SPY→$313) makes it worth $167

  The market's smooth skew can't easily embed these discrete jumps.
  That's where our model finds edge.
""")

    # ─── Part 4: How skew difference changes with recession prob ──
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 4: SKEW GAP AT DIFFERENT RECESSION PROBABILITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Focusing on the 480 Put (11% OTM):
""")

    K_focus = 480
    otm_dist = max(0, 1.0 - K_focus / S)
    market_vol = base_vol + otm_dist * 0.6 + otm_dist**2 * 1.5
    market_price = bs_price(S, K_focus, T, r, market_vol, OptionType.PUT)

    prob_rows = []
    for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.42, 0.50, 0.60, 0.70]:
        adj = blend_with_prediction_market(base, p)
        hv, _ = hybrid_price(S, K_focus, T, r, adj.scenarios, OptionType.PUT)
        hiv = implied_vol(hv, S, K_focus, T, r, OptionType.PUT)

        if hiv is not None:
            gap = hiv - market_vol
            gap_str = f"{gap:+.1%}"
            if gap > 0.01:
                verdict = "CHEAP — hybrid IV > market IV"
            elif gap < -0.01:
                verdict = "RICH — hybrid IV < market IV"
            else:
                verdict = "FAIR — skews roughly agree"
        else:
            gap_str = "N/A"
            verdict = ""

        marker = " ← Polymarket" if p == 0.42 else ""

        prob_rows.append({
            "P(recession)": f"{p:.0%}",
            "Market IV": f"{market_vol:.1%}",
            "Hybrid IV": f"{hiv:.1%}" if hiv else ">>100%",
            "IV Gap": gap_str,
            "Verdict": verdict + marker,
        })

    print(tabulate(prob_rows, headers="keys", tablefmt="simple"))

    print(f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUMMARY: WHAT THE SKEW DIFFERENCE TELLS YOU
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MARKET SKEW:
    - Smooth, continuous curve
    - Set by supply/demand for each strike
    - Embeds an average "fear premium"
    - Does NOT explicitly price discrete macro events

  HYBRID MODEL SKEW:
    - Built from discrete crash scenarios
    - Weighted by prediction market probabilities
    - Explicitly prices "what if recession?" at each severity level
    - Discontinuous — jumps at scenario boundaries

  THE GAP:
    When hybrid IV > market IV → the put is CHEAP
      → Market hasn't priced in as much crash risk as Polymarket implies
      → This is your signal

    When hybrid IV < market IV → the put is RICH
      → Market already prices more fear than scenarios justify
      → Avoid or sell

    The gap is largest for:
      - Deep OTM puts (where crash scenarios make them suddenly ITM)
      - Higher prediction market probabilities
      - Strikes near scenario shock levels (e.g. near SPY×0.88 for mild recession)
""")


if __name__ == "__main__":
    main()
