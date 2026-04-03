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

from src.hybrid_macro_vol.models import OptionType, ScenarioSet
from src.hybrid_macro_vol.pricing.scenarios import load_scenario_maps, blend_with_prediction_market
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

    # ─── Tab layout ───────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Scenario Assumptions",
        "Option Scan",
        "Trade Recommendations",
        "Sensitivity Analysis",
        "Backtest",
    ])

    # ─── Tab 1: Scenarios ─────────────────────────
    with tab1:
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

    # ─── Tab 2: Option Scan ───────────────────────
    with tab2:
        st.subheader(f"Mispriced Options Scan — {ticker}")

        with st.spinner("Scanning option chain..."):
            connector = OptionsDataConnector()
            chain = connector.get_option_chain(ticker)

            if chain is None or not chain.quotes:
                st.error("Could not fetch option chain data.")
            else:
                st.info(f"Underlying: ${chain.underlying_price:.2f} | "
                        f"Options loaded: {len(chain.quotes)} | "
                        f"Risk-free rate: {chain.risk_free_rate:.2%}")

                otype = OptionType.PUT if option_type == "PUT" else OptionType.CALL
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

    # ─── Tab 3: Trade Recommendations ─────────────
    with tab3:
        st.subheader("Recommended Trade Structures")

        if chain and chain.quotes:
            otype = OptionType.PUT if option_type == "PUT" else OptionType.CALL
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

    # ─── Tab 4: Sensitivity ───────────────────────
    with tab4:
        st.subheader("Sensitivity Analysis")
        st.caption("How does the edge change as event probability varies?")

        if chain and chain.quotes:
            # Pick a representative option (top signal or user-selected)
            otype = OptionType.PUT if option_type == "PUT" else OptionType.CALL
            put_quotes = [q for q in chain.quotes if q.option_type == otype and q.mid > 0.10]
            if put_quotes:
                strikes = sorted(set(q.strike for q in put_quotes))
                selected_strike = st.select_slider("Strike", options=strikes, value=strikes[len(strikes)//2])

                expiries = sorted(set(q.expiry for q in put_quotes if q.strike == selected_strike))
                if expiries:
                    selected_expiry = st.selectbox("Expiry", expiries)

                    # Find the quote
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

                        # Chart
                        chart_df = sens_df.copy()
                        chart_df["hybrid_value_num"] = chart_df["hybrid_value"]
                        chart_df["market_mid_num"] = chart_df["market_mid"]
                        st.line_chart(chart_df.set_index("event_prob")[["hybrid_value_num", "market_mid_num"]])
            else:
                st.warning("No options available for sensitivity analysis.")

    # ─── Tab 5: Backtest ──────────────────────────
    with tab5:
        st.subheader("Backtest Results (Synthetic Data)")
        st.caption("Tests whether prediction-market-informed option selection produces positive expected value")

        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                snapshots = generate_sample_snapshots(ticker=ticker)
                config = BacktestConfig(
                    base_scenario_set=base_scenarios,
                    option_type=OptionType.PUT if option_type == "PUT" else OptionType.CALL,
                    holding_period_days=30,
                    min_edge_score=0.3,
                    min_edge_pct=0.03,
                )
                result = run_backtest(snapshots, config)

                # Summary metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Total Trades", len(result.trades))
                col2.metric("Win Rate", f"{result.win_rate:.1%}")
                col3.metric("Avg Return", f"{result.avg_return:.1%}")
                col4.metric("Total P&L", f"${result.total_pnl:,.2f}")
                col5.metric("Max Drawdown", f"${result.max_drawdown:,.2f}")

                if result.sharpe:
                    st.metric("Sharpe Ratio", f"{result.sharpe:.2f}")

                # Trade details
                trades_df = backtest_results_to_dataframe(result)
                st.dataframe(trades_df, use_container_width=True, hide_index=True)

                # P&L by regime
                if result.by_regime:
                    st.subheader("P&L by Regime")
                    regime_df = pd.DataFrame([
                        {"Regime": k, "P&L": v}
                        for k, v in result.by_regime.items()
                    ])
                    st.bar_chart(regime_df.set_index("Regime"))


if __name__ == "__main__":
    main()
