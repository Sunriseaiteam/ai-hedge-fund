"""
Backtest Walkthrough
====================

Shows exactly how the backtest works, step by step, with a realistic
scenario that includes an actual drawdown so you can see wins AND losses.

The backtest answers: "If I had followed the hybrid model's signals
over the past year, would I have made money?"
"""

from __future__ import annotations

from datetime import date, timedelta
from tabulate import tabulate

from src.hybrid_macro_vol.models import OptionType, BacktestTrade
from src.hybrid_macro_vol.pricing.black_scholes import bs_price
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
from src.hybrid_macro_vol.pricing.hybrid_pricer import hybrid_price, scan_chain
from src.hybrid_macro_vol.signals.edge_scoring import generate_signal
from src.hybrid_macro_vol.connectors.options_data import OptionsDataConnector


def main():
    print("""
╔══════��═══════════════════════════════════════════════════════════════════╗
║  BACKTEST WALKTHROUGH: How the Historical Replay Works                 ║
╚══════════════════════════════════════════════════════════════════════════╝

  The backtest simulates 12 months of trading to test:
    1. Does the hybrid model find real edge?
    2. Do prediction-market-informed trades make money?

  We simulate a REALISTIC scenario: a recession scare that partially
  materializes, then recovers. Some months SPY drops, some it recovers.
""")

    maps = load_scenario_maps()
    base = maps["recession"]
    connector = OptionsDataConnector()
    r = 0.045

    # ─── The simulated history ────────────────────────────────────
    # A narrative: recession fears rise, SPY drops, then recovers
    months = [
        # (date, polymarket_prob, spy_price, description)
        ("2025-01", 0.15, 540, "Economy looks fine"),
        ("2025-02", 0.18, 535, "Weak jobs report"),
        ("2025-03", 0.25, 522, "Tariff fears emerge"),
        ("2025-04", 0.35, 505, "Manufacturing contraction"),
        ("2025-05", 0.45, 485, "Recession fears spike"),
        ("2025-06", 0.50, 470, "Two negative GDP prints"),
        ("2025-07", 0.42, 478, "Fed signals cuts"),
        ("2025-08", 0.32, 498, "Stimulus announced"),
        ("2025-09", 0.22, 518, "Recovery signs"),
        ("2025-10", 0.15, 535, "Soft landing narrative"),
        ("2025-11", 0.12, 545, "Strong earnings"),
        ("2025-12", 0.10, 555, "Rally continues"),
    ]

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("THE SIMULATED YEAR")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    history_rows = []
    for m, prob, price, desc in months:
        history_rows.append({
            "Month": m,
            "P(recession)": f"{prob:.0%}",
            "SPY": f"${price}",
            "Narrative": desc,
        })
    print(tabulate(history_rows, headers="keys", tablefmt="simple"))

    print(f"""

  SPY path: $540 → $470 (bottom) → $555 (recovery)
  Polymarket: 15% → 50% (peak fear) → 10% (fear fades)

  Now let's trace what the model does each month.
""")

    # ─── Month-by-month backtest ──────────────────────────────────
    all_trades: list[dict] = []
    cumulative_pnl = 0.0

    for i, (month, prob, spy, desc) in enumerate(months):
        # Can't trade last month (no realized price after it)
        if i >= len(months) - 1:
            break

        # Next month's price = our exit price (30-day hold)
        _, _, next_spy, _ = months[i + 1]

        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"MONTH: {month}  |  SPY=${spy}  |  Polymarket={prob:.0%}  |  {desc}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Step 1: Blend scenarios
        adjusted = blend_with_prediction_market(base, prob)
        print(f"\n  Step 1: Blend Polymarket {prob:.0%} into scenarios")
        for s in adjusted.scenarios:
            print(f"    {s.name:20s}  P={s.probability:5.1%}  shock={s.spot_shock:+.0%}")

        # Step 2: Generate synthetic option chain at current SPY
        chain = connector._generate_synthetic_chain("SPY", spot_override=float(spy))

        # Pick a specific target: 90-DTE put, ~10% OTM
        target_strike = round(spy * 0.90 / 5) * 5  # round to nearest 5
        T = 90 / 365.0

        # Get the market price (from synthetic chain's BS pricing)
        otm_dist = max(0, 1.0 - target_strike / spy)
        market_vol = 0.18 + otm_dist * 0.6 + otm_dist**2 * 1.5
        market_price = bs_price(spy, target_strike, T, r, market_vol, OptionType.PUT)

        # Step 3: Hybrid price
        hv, breakdown = hybrid_price(spy, target_strike, T, r, adjusted.scenarios, OptionType.PUT)
        edge = hv - market_price

        print(f"\n  Step 2: Price the {target_strike} Put (90 DTE, {otm_dist*100:.0f}% OTM)")
        print(f"    Market price (at {market_vol:.0%} IV): ${market_price:.2f}")
        print(f"    Hybrid price:                         ${hv:.2f}")
        print(f"    Edge:                                 ${edge:+.2f} ({edge/market_price*100:+.0f}%)")

        # Step 4: Decide whether to trade
        edge_pct = edge / market_price if market_price > 0 else 0
        trade_it = edge_pct > 0.10  # only trade if >10% edge

        print(f"\n  Step 3: Trade decision")
        if trade_it:
            print(f"    Edge {edge_pct:.0%} > 10% threshold → BUY")
        else:
            print(f"    Edge {edge_pct:.0%} < 10% threshold → SKIP")

        if not trade_it:
            print(f"    No trade this month.\n")
            continue

        # Step 5: What actually happened (30 days later)
        # Exit value = intrinsic value at next month's price
        exit_intrinsic = max(target_strike - next_spy, 0.0)
        pnl = exit_intrinsic - market_price
        pnl_pct = pnl / market_price if market_price > 0 else 0
        cumulative_pnl += pnl

        spy_move = (next_spy - spy) / spy
        put_itm = next_spy < target_strike

        print(f"\n  Step 4: What happened 30 days later")
        print(f"    SPY moved: ${spy} → ${next_spy} ({spy_move:+.1%})")
        print(f"    {target_strike} Put at expiry: {'ITM' if put_itm else 'OTM'}")
        print(f"    Exit value: ${exit_intrinsic:.2f} (intrinsic)")
        print(f"    P&L: ${pnl:+.2f} per share ({pnl_pct:+.0%})")

        if pnl > 0:
            print(f"    ✓ WIN — SPY dropped enough for the put to pay off")
        else:
            print(f"    ✗ LOSS — SPY didn't drop below {target_strike}, put expired worthless")

        print(f"    Cumulative P&L: ${cumulative_pnl:+.2f}\n")

        all_trades.append({
            "Month": month,
            "SPY": f"${spy}",
            "P(recession)": f"{prob:.0%}",
            "Strike": f"${target_strike}",
            "Entry $": f"${market_price:.2f}",
            "Hybrid $": f"${hv:.2f}",
            "Edge": f"{edge_pct:+.0%}",
            "SPY Next": f"${next_spy}",
            "SPY Move": f"{spy_move:+.1%}",
            "Exit $": f"${exit_intrinsic:.2f}",
            "P&L": f"${pnl:+.2f}",
            "Result": "WIN" if pnl > 0 else "LOSS",
        })

    # ─── Summary ──────────────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BACKTEST SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
    print(tabulate(all_trades, headers="keys", tablefmt="simple"))

    wins = sum(1 for t in all_trades if t["Result"] == "WIN")
    losses = sum(1 for t in all_trades if t["Result"] == "LOSS")
    total = len(all_trades)

    print(f"""
  Total trades:    {total}
  Wins:            {wins}
  Losses:          {losses}
  Win rate:        {wins/total:.0%} ({wins}/{total})
  Cumulative P&L:  ${cumulative_pnl:+.2f} per share
""")

    # ─── Explain the logic ────────────────────────────────────────
    print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW THE BACKTEST WORKS — STEP BY STEP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  For EACH month in the historical data:

  ┌─────────────────────────────────────────────────────────────┐
  │ 1. SNAPSHOT                                                  │
  │    Take a snapshot: SPY price + Polymarket recession prob    │
  │                                                              │
  │ 2. BLEND                                                     │
  │    Blend Polymarket prob into scenario weights               │
  │    e.g. 45% recession → 55% no-recession, 26% mild,        │
  │         13% severe, 6% crisis                                │
  │                                                              │
  │ 3. SCAN                                                      │
  │    Price every put in the chain under the hybrid model       │
  │    Compare hybrid value to market price → edge               │
  │                                                              │
  │ 4. FILTER                                                    │
  │    Only trade if edge > threshold (e.g. 10%)                 │
  │    Pick the top N signals by edge score                      │
  │                                                              │
  │ 5. HOLD                                                      │
  │    Hold the position for 30 days                             │
  │                                                              │
  │ 6. EXIT                                                      │
  │    Look up what SPY actually did 30 days later               │
  │    Compute intrinsic value of the put at that price          │
  │    P&L = exit_value - entry_price                            │
  │                                                              │
  │ 7. RECORD                                                    │
  │    Log the trade: entry, exit, P&L, what regime it was       │
  └─────────────────────────────────────────────────────────────┘

  After all months are processed:

  ┌─────────────────────────────────────────────────────────────┐
  │ AGGREGATE                                                    │
  │ • Win rate                                                   │
  │ • Average return per trade                                   │
  │ • Total P&L                                                  │
  │ • Max drawdown                                               │
  │ • Sharpe ratio                                               │
  │ • P&L by regime (when did the model make/lose money?)        │
  │ • P&L by event probability bucket                            │
  └─────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THE BACKTEST IS TESTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Two separate hypotheses:

  1. "Prediction markets add information"
     If Polymarket says 45% recession and the options market is only
     pricing 10%, is the 35% gap a real signal or noise?
     → Test by measuring P&L in months where the gap is large vs small

  2. "Our scenario mapping is accurate"
     When we say "mild recession = SPY -12%", is that calibrated?
     → Test by comparing predicted scenario outcomes to actual SPY moves
     → Calibration error metric


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT LIMITATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Right now the backtest uses SYNTHETIC data (fake option chains
  and a made-up SPY price path). To make it real:

  1. Need historical Polymarket/Kalshi probabilities over time
     → Scrape or buy historical prediction market data

  2. Need historical option chains (full strikes + expiries)
     → CBOE data, OptionMetrics, or similar provider

  3. Need actual SPY prices (already available via existing API)

  4. Ideally: historical VIX to calibrate the synthetic chain's
     base vol to what markets were actually pricing

  With real data, you'd see:
  • Whether the model's "cheap put" signals actually paid off
  • Whether large Polymarket-vs-options gaps predicted drawdowns
  • Which strike/expiry selection rules work best
  • Whether the edge is captured or eaten by slippage/spreads
""")


if __name__ == "__main__":
    main()
