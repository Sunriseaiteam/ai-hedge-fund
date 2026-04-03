"""
Main runner script for the Hybrid Macro-Options Signal Engine.

Runs a full scan: loads scenarios, fetches/generates option data,
blends prediction market probabilities, prices everything,
and outputs ranked trade recommendations.

Usage:
    python -m src.hybrid_macro_vol.run_scan
    python -m src.hybrid_macro_vol.run_scan --ticker SPY --event recession --prob 0.35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date

from tabulate import tabulate

from src.hybrid_macro_vol.models import OptionType
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
from src.hybrid_macro_vol.connectors.options_data import OptionsDataConnector
from src.hybrid_macro_vol.signals.ranking import rank_opportunities, signals_to_dataframe
from src.hybrid_macro_vol.signals.trade_construction import recommend_trades


def run_scan(
    ticker: str = "SPY",
    event_type: str = "recession",
    event_probability: float = 0.30,
    option_type: str = "put",
    top_n: int = 15,
) -> None:
    """Run a full mispriced-options scan."""
    print(f"\n{'='*80}")
    print(f"  HYBRID MACRO-OPTIONS SIGNAL ENGINE")
    print(f"  Scanning {ticker} {option_type}s | Event: {event_type} | P(event) = {event_probability:.0%}")
    print(f"{'='*80}\n")

    # 1. Load scenario maps
    scenario_maps = load_scenario_maps()
    if event_type not in scenario_maps:
        print(f"Unknown event type '{event_type}'. Available: {list(scenario_maps.keys())}")
        sys.exit(1)

    base_scenarios = scenario_maps[event_type]
    print(f"Base scenarios: {base_scenarios.name}")
    for s in base_scenarios.scenarios:
        print(f"  {s.name:20s}  P={s.probability:.0%}  spot={s.spot_shock:+.0%}  vol={s.volatility:.0%}")

    # 2. Blend with prediction market probability
    adjusted = blend_with_prediction_market(base_scenarios, event_probability)
    print(f"\nAdjusted for P(event) = {event_probability:.0%}:")
    for s in adjusted.scenarios:
        print(f"  {s.name:20s}  P={s.probability:.0%}  spot={s.spot_shock:+.0%}  vol={s.volatility:.0%}")

    # 3. Fetch option chain
    print(f"\nLoading option chain for {ticker}...")
    connector = OptionsDataConnector()
    chain = connector.get_option_chain(ticker)
    if chain is None:
        print("Failed to load option chain.")
        sys.exit(1)

    otype = OptionType.PUT if option_type.lower() == "put" else OptionType.CALL
    n_opts = sum(1 for q in chain.quotes if q.option_type == otype)
    print(f"  Underlying: ${chain.underlying_price:.2f}")
    print(f"  {option_type.upper()} options loaded: {n_opts}")

    # 4. Scan and rank
    print(f"\nScanning for mispriced {option_type}s...")
    signals = rank_opportunities(
        chain=chain,
        scenario_set=adjusted,
        option_type=otype,
        top_n=top_n,
    )

    if not signals:
        print("No signals found.")
        return

    # 5. Display results
    df = signals_to_dataframe(signals)
    print(f"\n{'─'*80}")
    print(f"  TOP {len(signals)} MISPRICED OPTIONS")
    print(f"{'─'*80}")
    print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    # 6. Trade recommendations
    print(f"\n{'─'*80}")
    print(f"  TRADE RECOMMENDATIONS")
    print(f"{'─'*80}")
    recs = recommend_trades(
        signals=signals,
        chain_quotes=chain.quotes,
        scenario_set=adjusted,
        risk_free_rate=chain.risk_free_rate,
    )

    for i, rec in enumerate(recs):
        print(f"\n  #{i+1}: {rec.name}")
        print(f"  Edge Score: {rec.edge_score:.2f}")
        print(f"  Net Debit: ${rec.net_debit:.2f}/share" if rec.net_debit else "")
        print(f"  Max Loss: ${rec.max_loss:.0f}/contract" if rec.max_loss else "")
        print(f"  Max Gain: {'${:.0f}'.format(rec.max_gain) if rec.max_gain else 'Unlimited'}/contract")
        print(f"  Breakeven: ${rec.breakeven:.0f}" if rec.breakeven else "")
        print(f"  Rationale: {rec.rationale}")
        if rec.scenario_pnl:
            print(f"  Scenario P&L:")
            for sc_name, pnl in rec.scenario_pnl.items():
                marker = "+" if pnl > 0 else ""
                print(f"    {sc_name:20s}: {marker}${pnl:,.0f}")

    print(f"\n{'='*80}")
    print(f"  Scan complete. {len(signals)} signals, {len(recs)} recommendations.")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Macro-Options Signal Engine")
    parser.add_argument("--ticker", default="SPY", help="Ticker to scan (default: SPY)")
    parser.add_argument("--event", default="recession", help="Event type (recession, fed_rate_cut, tariff_escalation, government_shutdown)")
    parser.add_argument("--prob", type=float, default=0.30, help="Event probability from prediction markets (0-1)")
    parser.add_argument("--type", default="put", choices=["put", "call"], help="Option type to scan")
    parser.add_argument("--top", type=int, default=15, help="Number of top results")
    args = parser.parse_args()

    run_scan(
        ticker=args.ticker,
        event_type=args.event,
        event_probability=args.prob,
        option_type=args.type,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
