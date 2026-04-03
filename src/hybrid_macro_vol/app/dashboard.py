"""
Streamlit dashboard for the Hybrid Macro-Options Signal Engine.

Run with: streamlit run src/hybrid_macro_vol/app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import streamlit as st
import pandas as pd
import numpy as np

from src.hybrid_macro_vol.models import OptionType, ScenarioSet
from src.hybrid_macro_vol.pricing.black_scholes import bs_price, implied_vol
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
from src.hybrid_macro_vol.pricing.hybrid_pricer import hybrid_price, compute_breakeven_probability
from src.hybrid_macro_vol.connectors.options_data import OptionsDataConnector
from src.hybrid_macro_vol.signals.ranking import rank_opportunities, signals_to_dataframe, sensitivity_table
from src.hybrid_macro_vol.signals.trade_construction import recommend_trades
from src.hybrid_macro_vol.backtest.replay import (
    BacktestConfig,
    generate_sample_snapshots,
    run_backtest,
    backtest_results_to_dataframe,
)


def main():
    st.set_page_config(page_title="Macro-Vol Signal Engine", layout="wide")
    st.title("Hybrid Macro-Options Signal Engine")
    st.caption("Find mispriced options by comparing prediction market probabilities to options pricing")

    # ─── Sidebar: Inputs ──────────────────────────
    with st.sidebar:
        st.header("Configuration")

        ticker = st.selectbox("Ticker", ["SPY", "QQQ", "IWM", "SPX"], index=0)

        scenario_maps = load_scenario_maps()
        event_type = st.selectbox(
            "Event Type",
            list(scenario_maps.keys()),
            format_func=lambda k: scenario_maps[k].name,
        )
        base_scenarios = scenario_maps[event_type]

        st.subheader("Prediction Market Input")
        event_prob = st.slider(
            "Event Probability (from prediction markets)",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.01,
            help="Set this to the current probability from Polymarket/Kalshi",
        )

        st.subheader("Filters")
        option_type = st.selectbox("Option Type", ["PUT", "CALL"], index=0)
        min_dte = st.number_input("Min DTE", value=14, min_value=1)
        max_dte = st.number_input("Max DTE", value=365, min_value=30)
        top_n = st.number_input("Top N Results", value=20, min_value=5, max_value=100)

    # Blend scenarios with prediction market probability
    adjusted_scenarios = blend_with_prediction_market(base_scenarios, event_prob)

    # Load chain once, reuse across tabs
    connector = OptionsDataConnector()
    chain = connector.get_option_chain(ticker)
    otype = OptionType.PUT if option_type == "PUT" else OptionType.CALL

    # ─── Tab layout ───────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Skew Difference",
        "Scenario Assumptions",
        "Option Scan",
        "Trade Recommendations",
        "Sensitivity Analysis",
        "Backtest",
    ])

    # ─── Tab 1: Skew Difference (NEW — main view) ─
    with tab1:
        st.subheader("Vol Skew: Market vs Hybrid Model")
        st.markdown(
            f"Comparing the market's implied vol curve against what our scenario model "
            f"implies, given **{event_prob:.0%}** event probability from prediction markets."
        )

        if chain is None or not chain.quotes:
            st.error("Could not load option chain.")
        else:
            S = chain.underlying_price
            r = chain.risk_free_rate

            # Let user pick an expiry for the skew comparison
            all_expiries = sorted(set(
                q.expiry for q in chain.quotes
                if q.option_type == otype and q.mid > 0.01
            ))
            if not all_expiries:
                st.warning("No options available.")
            else:
                selected_expiry = st.selectbox(
                    "Expiry for skew comparison",
                    all_expiries,
                    index=min(3, len(all_expiries) - 1),
                    format_func=lambda d: f"{d.isoformat()} ({(d - pd.Timestamp.now().date()).days} DTE)",
                )
                from datetime import date
                T = max((selected_expiry - date.today()).days / 365.0, 0.001)

                # Gather all strikes for this expiry
                expiry_quotes = sorted(
                    [q for q in chain.quotes
                     if q.expiry == selected_expiry and q.option_type == otype and q.mid > 0.01],
                    key=lambda q: q.strike,
                )

                if not expiry_quotes:
                    st.warning("No quotes for this expiry.")
                else:
                    strikes = [q.strike for q in expiry_quotes]
                    market_ivs = [q.implied_vol or 0.20 for q in expiry_quotes]
                    market_prices = [q.mid for q in expiry_quotes]

                    # Compute hybrid price and implied vol for each strike
                    hybrid_ivs = []
                    hybrid_prices = []
                    edges = []
                    breakevens = []
                    scenario_decomp = []

                    for q in expiry_quotes:
                        hv, breakdown = hybrid_price(
                            S, q.strike, T, r, adjusted_scenarios.scenarios, otype,
                        )
                        hybrid_prices.append(hv)
                        edges.append(hv - q.mid)

                        hiv = implied_vol(hv, S, q.strike, T, r, otype)
                        hybrid_ivs.append(hiv if hiv else None)

                        be = compute_breakeven_probability(
                            S, q.strike, T, r, q.mid, adjusted_scenarios, otype,
                        )
                        breakevens.append(be)
                        scenario_decomp.append(breakdown)

                    # ── Chart 1: Vol Skew Comparison ──────────
                    st.markdown("### Implied Volatility Skew")
                    st.markdown(
                        "**Market IV** = what the options market charges. "
                        "**Hybrid IV** = what the model says it should be. "
                        "Gap = potential mispricing."
                    )

                    skew_df = pd.DataFrame({
                        "Strike": strikes,
                        "Market IV": [iv * 100 for iv in market_ivs],
                        "Hybrid IV": [(iv * 100 if iv else None) for iv in hybrid_ivs],
                    }).set_index("Strike")
                    st.line_chart(skew_df, color=["#4488ff", "#ff4444"])

                    # ── Chart 2: IV Gap (the edge in vol terms) ──
                    st.markdown("### IV Gap (Hybrid IV - Market IV)")
                    st.markdown(
                        "Positive = put is **cheap** (hybrid model wants higher vol). "
                        "Negative = put is **rich**."
                    )

                    iv_gaps = []
                    for miv, hiv in zip(market_ivs, hybrid_ivs):
                        if hiv is not None:
                            iv_gaps.append((hiv - miv) * 100)
                        else:
                            iv_gaps.append(None)

                    gap_df = pd.DataFrame({
                        "Strike": strikes,
                        "IV Gap (pts)": iv_gaps,
                    }).set_index("Strike")
                    st.bar_chart(gap_df)

                    # ── Chart 3: Dollar Edge ─────────────────
                    st.markdown("### Dollar Edge per Strike")
                    st.markdown("Hybrid fair value minus market mid price.")

                    edge_df = pd.DataFrame({
                        "Strike": strikes,
                        "Edge ($)": [round(e, 2) for e in edges],
                    }).set_index("Strike")
                    st.bar_chart(edge_df)

                    # ── Chart 4: Breakeven probability ────────
                    st.markdown("### Break-Even Event Probability")
                    st.markdown(
                        f"What event probability justifies the current market price? "
                        f"The red line is Polymarket's {event_prob:.0%}."
                    )

                    be_df = pd.DataFrame({
                        "Strike": strikes,
                        "Break-Even Prob (%)": [(b * 100 if b else None) for b in breakevens],
                        "Polymarket (%)": [event_prob * 100] * len(strikes),
                    }).set_index("Strike")
                    st.line_chart(be_df, color=["#4488ff", "#ff4444"])

                    # ── Table: Full Detail ────────────────────
                    st.markdown("### Strike-by-Strike Detail")

                    detail_rows = []
                    for i, q in enumerate(expiry_quotes):
                        miv = market_ivs[i]
                        hiv = hybrid_ivs[i]
                        iv_gap = (hiv - miv) if hiv else None

                        if iv_gap is not None and iv_gap > 0.01:
                            verdict = "CHEAP"
                        elif iv_gap is not None and iv_gap < -0.01:
                            verdict = "RICH"
                        else:
                            verdict = "FAIR"

                        detail_rows.append({
                            "Strike": f"${q.strike:.0f}",
                            "OTM %": f"{(1 - q.strike/S)*100:.0f}%" if otype == OptionType.PUT else f"{(q.strike/S - 1)*100:.0f}%",
                            "Market IV": f"{miv:.1%}",
                            "Hybrid IV": f"{hiv:.1%}" if hiv else ">100%",
                            "IV Gap": f"{iv_gap:+.1%}" if iv_gap else "N/A",
                            "Market $": f"${q.mid:.2f}",
                            "Hybrid $": f"${hybrid_prices[i]:.2f}",
                            "Edge $": f"${edges[i]:+.2f}",
                            "Break-Even": f"{breakevens[i]:.1%}" if breakevens[i] else "N/A",
                            "Verdict": verdict,
                        })

                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

                    # ── Scenario Decomposition for selected strike ──
                    st.markdown("### Scenario Value Decomposition")
                    st.markdown("Pick a strike to see how each scenario contributes to the hybrid price.")

                    decomp_strike = st.select_slider(
                        "Strike for decomposition",
                        options=strikes,
                        value=strikes[len(strikes) // 3],  # default to ~OTM area
                        key="decomp_strike",
                    )

                    # Find the matching data
                    idx = next(
                        (i for i, q in enumerate(expiry_quotes) if q.strike == decomp_strike),
                        None,
                    )
                    if idx is not None:
                        q = expiry_quotes[idx]
                        breakdown = scenario_decomp[idx]
                        hv_total = hybrid_prices[idx]

                        col_left, col_right = st.columns([2, 1])

                        with col_left:
                            decomp_rows = []
                            chart_scenarios = []
                            chart_weighted = []
                            for s in adjusted_scenarios.scenarios:
                                val = breakdown[s.name]
                                weighted = s.probability * val
                                pct = (weighted / hv_total * 100) if hv_total > 0 else 0
                                shocked_spot = S * (1 + s.spot_shock)
                                itm_label = "ITM" if (otype == OptionType.PUT and shocked_spot < q.strike) or (otype == OptionType.CALL and shocked_spot > q.strike) else "OTM"

                                decomp_rows.append({
                                    "Scenario": s.name,
                                    "Prob": f"{s.probability:.1%}",
                                    f"{ticker} goes to": f"${shocked_spot:.0f}",
                                    "ITM/OTM": itm_label,
                                    "Option Value": f"${val:.2f}",
                                    "Weighted": f"${weighted:.2f}",
                                    "% of Total": f"{pct:.1f}%",
                                })
                                chart_scenarios.append(s.name)
                                chart_weighted.append(weighted)

                            st.dataframe(pd.DataFrame(decomp_rows), use_container_width=True, hide_index=True)

                            # Bar chart of weighted contributions
                            contrib_df = pd.DataFrame({
                                "Scenario": chart_scenarios,
                                "Weighted Value ($)": chart_weighted,
                            }).set_index("Scenario")
                            st.bar_chart(contrib_df)

                        with col_right:
                            st.metric("Strike", f"${decomp_strike:.0f}")
                            st.metric("Market Price", f"${q.mid:.2f}")
                            st.metric("Hybrid Price", f"${hv_total:.2f}")
                            st.metric("Edge", f"${edges[idx]:+.2f}")
                            miv = market_ivs[idx]
                            hiv = hybrid_ivs[idx]
                            st.metric("Market IV", f"{miv:.1%}")
                            st.metric("Hybrid IV", f"{hiv:.1%}" if hiv else ">100%")
                            if hiv:
                                st.metric("IV Gap", f"{(hiv-miv)*100:+.1f} pts")
                            if breakevens[idx]:
                                st.metric("Break-Even Prob", f"{breakevens[idx]:.1%}")
                                if breakevens[idx] < event_prob:
                                    st.success(f"Polymarket ({event_prob:.0%}) > Break-even ({breakevens[idx]:.1%}) = CHEAP")
                                else:
                                    st.warning(f"Polymarket ({event_prob:.0%}) < Break-even ({breakevens[idx]:.1%}) = RICH")

    # ─── Tab 2: Scenarios ─────────────────────────
    with tab2:
        st.subheader("Scenario Weights")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Base Scenarios (default)**")
            base_df = pd.DataFrame([
                {
                    "Scenario": s.name,
                    "Probability": f"{s.probability:.1%}",
                    "Spot Shock": f"{s.spot_shock:+.1%}",
                    "Volatility": f"{s.volatility:.0%}",
                    "Description": s.description,
                }
                for s in base_scenarios.scenarios
            ])
            st.dataframe(base_df, use_container_width=True, hide_index=True)

        with col2:
            st.markdown(f"**Adjusted Scenarios (event prob = {event_prob:.0%})**")
            adj_df = pd.DataFrame([
                {
                    "Scenario": s.name,
                    "Probability": f"{s.probability:.1%}",
                    "Spot Shock": f"{s.spot_shock:+.1%}",
                    "Volatility": f"{s.volatility:.0%}",
                    "Description": s.description,
                }
                for s in adjusted_scenarios.scenarios
            ])
            st.dataframe(adj_df, use_container_width=True, hide_index=True)

        # Probability comparison chart
        chart_data = pd.DataFrame({
            "Scenario": [s.name for s in base_scenarios.scenarios],
            "Base": [s.probability for s in base_scenarios.scenarios],
            "Adjusted": [s.probability for s in adjusted_scenarios.scenarios],
        })
        st.bar_chart(chart_data.set_index("Scenario"))

    # ─── Tab 3: Option Scan ───────────────────────
    with tab3:
        st.subheader(f"Mispriced Options Scan — {ticker}")

        if chain is None or not chain.quotes:
            st.error("Could not fetch option chain data.")
        else:
            st.info(f"Underlying: ${chain.underlying_price:.2f} | "
                    f"Options loaded: {len(chain.quotes)} | "
                    f"Risk-free rate: {chain.risk_free_rate:.2%}")

            signals = rank_opportunities(
                chain=chain,
                scenario_set=adjusted_scenarios,
                option_type=otype,
                min_dte=min_dte,
                max_dte=max_dte,
                top_n=top_n,
            )

            if not signals:
                st.warning("No signals found with current filters.")
            else:
                df = signals_to_dataframe(signals)
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Edge distribution
                st.subheader("Edge Distribution")
                edge_data = pd.DataFrame({
                    "strike": [s.pricing.strike for s in signals],
                    "edge": [s.pricing.edge for s in signals],
                    "edge_score": [s.edge_score for s in signals],
                })
                st.bar_chart(edge_data.set_index("strike")["edge"])

    # ─── Tab 4: Trade Recommendations ─────────────
    with tab4:
        st.subheader("Recommended Trade Structures")

        if chain and chain.quotes:
            signals = rank_opportunities(
                chain=chain,
                scenario_set=adjusted_scenarios,
                option_type=otype,
                min_dte=min_dte,
                max_dte=max_dte,
                top_n=10,
            )

            if signals:
                recs = recommend_trades(
                    signals=signals,
                    chain_quotes=chain.quotes,
                    scenario_set=adjusted_scenarios,
                    risk_free_rate=chain.risk_free_rate,
                )

                for i, rec in enumerate(recs):
                    with st.expander(f"#{i+1}: {rec.name} (edge score: {rec.edge_score:.2f})", expanded=(i == 0)):
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Net Debit", f"${rec.net_debit:.2f}" if rec.net_debit else "N/A")
                        col2.metric("Max Loss", f"${rec.max_loss:.0f}" if rec.max_loss else "N/A")
                        col3.metric("Max Gain", f"${rec.max_gain:.0f}" if rec.max_gain else "Unlimited")
                        col4.metric("Breakeven", f"${rec.breakeven:.0f}" if rec.breakeven else "N/A")

                        st.markdown(f"**Rationale:** {rec.rationale}")

                        # Scenario P&L
                        if rec.scenario_pnl:
                            st.markdown("**Scenario P&L (per contract):**")
                            pnl_df = pd.DataFrame([
                                {"Scenario": k, "P&L": f"${v:,.0f}"}
                                for k, v in rec.scenario_pnl.items()
                            ])
                            st.dataframe(pnl_df, use_container_width=True, hide_index=True)

                        # Legs
                        st.markdown("**Legs:**")
                        for leg in rec.legs:
                            st.text(f"  {leg.action.value.upper()} {leg.quantity}x "
                                    f"{leg.strike:.0f} {leg.option_type.value} "
                                    f"@ ${leg.price:.2f}")
            else:
                st.warning("No signals to recommend trades from.")

    # ─── Tab 5: Sensitivity ───────────────────────
    with tab5:
        st.subheader("Sensitivity Analysis")
        st.caption("How does the edge change as event probability varies?")

        if chain and chain.quotes:
            put_quotes = [q for q in chain.quotes if q.option_type == otype and q.mid > 0.10]
            if put_quotes:
                strikes = sorted(set(q.strike for q in put_quotes))
                selected_strike = st.select_slider("Strike", options=strikes, value=strikes[len(strikes)//2], key="sens_strike")

                expiries = sorted(set(q.expiry for q in put_quotes if q.strike == selected_strike))
                if expiries:
                    selected_expiry = st.selectbox("Expiry", expiries, key="sens_expiry")

                    quote = next(
                        (q for q in put_quotes
                         if q.strike == selected_strike and q.expiry == selected_expiry),
                        None,
                    )
                    if quote:
                        probs = [i / 20.0 for i in range(1, 20)]
                        sens_df = sensitivity_table(
                            quote=quote,
                            underlying_price=chain.underlying_price,
                            risk_free_rate=chain.risk_free_rate,
                            scenario_set=base_scenarios,
                            prob_range=probs,
                        )
                        st.dataframe(sens_df, use_container_width=True, hide_index=True)

                        chart_df = sens_df.copy()
                        chart_df["hybrid_value_num"] = chart_df["hybrid_value"]
                        chart_df["market_mid_num"] = chart_df["market_mid"]
                        st.line_chart(chart_df.set_index("event_prob")[["hybrid_value_num", "market_mid_num"]])
            else:
                st.warning("No options available for sensitivity analysis.")

    # ─── Tab 6: Backtest ──────────────────────────
    with tab6:
        st.subheader("Backtest Results (Synthetic Data)")
        st.caption("Tests whether prediction-market-informed option selection produces positive expected value")

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                snapshots = generate_sample_snapshots(ticker=ticker)
                config = BacktestConfig(
                    base_scenario_set=base_scenarios,
                    option_type=otype,
                    holding_period_days=30,
                    min_edge_score=0.3,
                    min_edge_pct=0.03,
                )
                result = run_backtest(snapshots, config)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Trades", len(result.trades))
                col2.metric("Win Rate", f"{result.win_rate:.1%}")
                col3.metric("Avg Return", f"{result.avg_return:.1%}")
                col4.metric("Total P&L", f"${result.total_pnl:,.2f}")
                col5.metric("Max Drawdown", f"${result.max_drawdown:,.2f}")

                if result.sharpe:
                    st.metric("Sharpe Ratio", f"{result.sharpe:.2f}")

                trades_df = backtest_results_to_dataframe(result)
                st.dataframe(trades_df, use_container_width=True, hide_index=True)

                if result.by_regime:
                    st.subheader("P&L by Regime")
                    regime_df = pd.DataFrame([
                        {"Regime": k, "P&L": v}
                        for k, v in result.by_regime.items()
                    ])
                    st.bar_chart(regime_df.set_index("Regime"))


if __name__ == "__main__":
    main()
