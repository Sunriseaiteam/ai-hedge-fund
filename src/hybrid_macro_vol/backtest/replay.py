"""
Backtest replay engine.

Replays historical scenarios to test whether the hybrid model's signals
produce positive expected value over time.

Two things are being tested:
1. Whether prediction markets add information beyond options pricing
2. Whether our event → asset-price mapping is good
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from src.hybrid_macro_vol.models import (
    BacktestResult,
    BacktestTrade,
    OptionChain,
    OptionQuote,
    OptionType,
    ScenarioSet,
)
from src.hybrid_macro_vol.pricing.hybrid_pricer import scan_chain
from src.hybrid_macro_vol.pricing.scenarios import blend_with_prediction_market
from src.hybrid_macro_vol.signals.edge_scoring import generate_signal


@dataclass
class HistoricalSnapshot:
    """A single point-in-time snapshot for backtesting."""
    as_of: date
    event_probability: float  # prediction market prob at that date
    underlying_price: float
    option_chain: OptionChain
    realized_price_30d: float | None = None  # actual price 30 days later
    realized_price_60d: float | None = None
    realized_price_90d: float | None = None


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    base_scenario_set: ScenarioSet
    option_type: OptionType = OptionType.PUT
    holding_period_days: int = 30
    min_edge_score: float = 0.5
    min_edge_pct: float = 0.05
    max_positions: int = 5
    risk_free_rate: float = 0.045


def run_backtest(
    snapshots: Sequence[HistoricalSnapshot],
    config: BacktestConfig,
) -> BacktestResult:
    """
    Run a backtest across historical snapshots.

    For each snapshot:
    1. Blend prediction market probability into scenarios
    2. Scan the option chain for mispriced options
    3. Select top signals above threshold
    4. Compute P&L based on realized price movement

    This tests whether the divergence between prediction markets
    and options pricing is actually a tradable signal.
    """
    trades: list[BacktestTrade] = []

    for snap in snapshots:
        # Blend prediction market prob into scenario weights
        adjusted_scenarios = blend_with_prediction_market(
            config.base_scenario_set,
            snap.event_probability,
        )

        # Scan the chain
        results = scan_chain(
            chain=snap.option_chain.quotes,
            underlying_price=snap.underlying_price,
            risk_free_rate=config.risk_free_rate,
            scenario_set=adjusted_scenarios,
            option_type=config.option_type,
            min_mid=0.10,
        )

        # Generate and filter signals
        signals = [generate_signal(r) for r in results]
        signals = [
            s for s in signals
            if s.edge_score >= config.min_edge_score
            and s.pricing.edge_pct >= config.min_edge_pct
        ]
        signals.sort(key=lambda s: s.edge_score, reverse=True)
        top_signals = signals[:config.max_positions]

        # Compute realized P&L
        exit_date = snap.as_of + timedelta(days=config.holding_period_days)

        # Determine realized price at holding period end
        if config.holding_period_days <= 30:
            realized = snap.realized_price_30d
        elif config.holding_period_days <= 60:
            realized = snap.realized_price_60d
        else:
            realized = snap.realized_price_90d

        for sig in top_signals:
            p = sig.pricing
            entry_price = p.market_mid

            # Estimate exit value: intrinsic value at realized price
            # (simplified — ignores remaining time value)
            if realized is not None:
                if p.option_type == OptionType.PUT:
                    exit_value = max(p.strike - realized, 0.0)
                else:
                    exit_value = max(realized - p.strike, 0.0)
                pnl = exit_value - entry_price
                pnl_pct = pnl / entry_price if entry_price > 0 else 0.0
            else:
                exit_value = None
                pnl = None
                pnl_pct = None

            trades.append(BacktestTrade(
                entry_date=snap.as_of,
                exit_date=exit_date if realized else None,
                ticker=p.ticker,
                structure=f"Long {p.strike:.0f} {p.option_type.value}",
                entry_price=entry_price,
                exit_price=exit_value,
                pnl=pnl,
                pnl_pct=pnl_pct,
                scenario_at_entry=_dominant_scenario(adjusted_scenarios),
                event_prob_at_entry=snap.event_probability,
            ))

    return _compute_results(trades)


def _dominant_scenario(ss: ScenarioSet) -> str:
    """Return the scenario with highest probability."""
    if not ss.scenarios:
        return "unknown"
    return max(ss.scenarios, key=lambda s: s.probability).name


def _compute_results(trades: list[BacktestTrade]) -> BacktestResult:
    """Aggregate trade-level results into a BacktestResult."""
    completed = [t for t in trades if t.pnl is not None]

    if not completed:
        return BacktestResult(trades=trades)

    pnls = [t.pnl for t in completed]
    total_pnl = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(completed)
    avg_return = sum(t.pnl_pct for t in completed if t.pnl_pct is not None) / len(completed)

    # Max drawdown (cumulative)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    # Simple Sharpe approximation
    import numpy as np
    returns = [t.pnl_pct for t in completed if t.pnl_pct is not None]
    if len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(12) if np.std(returns) > 0 else None
    else:
        sharpe = None

    # P&L by regime
    by_regime: dict[str, float] = {}
    for t in completed:
        regime = t.scenario_at_entry or "unknown"
        by_regime[regime] = by_regime.get(regime, 0.0) + (t.pnl or 0.0)

    return BacktestResult(
        trades=trades,
        total_pnl=round(total_pnl, 2),
        win_rate=round(win_rate, 4),
        avg_return=round(avg_return, 4),
        max_drawdown=round(max_dd, 2),
        sharpe=round(sharpe, 2) if sharpe else None,
        by_regime={k: round(v, 2) for k, v in by_regime.items()},
    )


def generate_sample_snapshots(
    ticker: str = "SPY",
    n_snapshots: int = 12,
) -> list[HistoricalSnapshot]:
    """
    Generate synthetic historical snapshots for testing the backtest framework.

    Simulates a year of monthly snapshots with varying recession probabilities
    and corresponding price moves.
    """
    from src.hybrid_macro_vol.connectors.options_data import OptionsDataConnector

    connector = OptionsDataConnector()
    base_price = 540.0
    snapshots = []

    # Simulate monthly snapshots with a recession scare narrative
    prob_path = [0.15, 0.18, 0.22, 0.28, 0.35, 0.40, 0.38, 0.30, 0.25, 0.20, 0.15, 0.12]
    price_path = [540, 535, 528, 515, 500, 490, 495, 510, 525, 535, 545, 555]

    for i in range(min(n_snapshots, len(prob_path))):
        snap_date = date(2025, 1 + i, 15) if i < 12 else date(2026, i - 11, 15)
        spot = price_path[i]

        chain = connector._generate_synthetic_chain(ticker, spot_override=float(spot))

        realized_30d = price_path[i + 1] if i + 1 < len(price_path) else None
        realized_60d = price_path[i + 2] if i + 2 < len(price_path) else None
        realized_90d = price_path[i + 3] if i + 3 < len(price_path) else None

        snapshots.append(HistoricalSnapshot(
            as_of=snap_date,
            event_probability=prob_path[i],
            underlying_price=float(spot),
            option_chain=chain,
            realized_price_30d=float(realized_30d) if realized_30d else None,
            realized_price_60d=float(realized_60d) if realized_60d else None,
            realized_price_90d=float(realized_90d) if realized_90d else None,
        ))

    return snapshots


def backtest_results_to_dataframe(result: BacktestResult) -> pd.DataFrame:
    """Convert backtest trades to a DataFrame for analysis."""
    rows = []
    for t in result.trades:
        rows.append({
            "entry_date": t.entry_date.isoformat(),
            "exit_date": t.exit_date.isoformat() if t.exit_date else "N/A",
            "ticker": t.ticker,
            "structure": t.structure,
            "entry_price": round(t.entry_price, 2),
            "exit_price": round(t.exit_price, 2) if t.exit_price is not None else "N/A",
            "pnl": round(t.pnl, 2) if t.pnl is not None else "N/A",
            "pnl_pct": f"{t.pnl_pct:.1%}" if t.pnl_pct is not None else "N/A",
            "event_prob": f"{t.event_prob_at_entry:.0%}" if t.event_prob_at_entry else "N/A",
            "regime": t.scenario_at_entry,
        })
    return pd.DataFrame(rows)
