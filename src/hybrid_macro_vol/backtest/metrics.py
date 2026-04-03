"""Backtest performance metrics and analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.hybrid_macro_vol.models import BacktestResult, BacktestTrade


def compute_sharpe(returns: Sequence[float], periods_per_year: float = 12.0) -> float | None:
    """Annualized Sharpe ratio assuming zero risk-free for simplicity."""
    if len(returns) < 2:
        return None
    arr = np.array(returns)
    if np.std(arr) == 0:
        return None
    return float(np.mean(arr) / np.std(arr) * np.sqrt(periods_per_year))


def compute_max_drawdown(pnl_series: Sequence[float]) -> float:
    """Max drawdown from a series of cumulative P&L values."""
    cumulative = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0


def compute_calmar_ratio(
    total_return: float,
    max_drawdown: float,
    periods: float = 1.0,
) -> float | None:
    """Calmar ratio = annualized return / max drawdown."""
    if max_drawdown <= 0:
        return None
    annual_return = total_return / periods
    return annual_return / max_drawdown


def pnl_by_event_probability(
    trades: Sequence[BacktestTrade],
    buckets: Sequence[float] = (0.0, 0.20, 0.35, 0.50, 0.75, 1.0),
) -> dict[str, dict]:
    """
    Analyze P&L bucketed by the prediction market probability at entry.

    This tells us: does the signal work better when prediction markets
    show higher divergence from options pricing?
    """
    results: dict[str, dict] = {}
    for i in range(len(buckets) - 1):
        lo, hi = buckets[i], buckets[i + 1]
        label = f"{lo:.0%}-{hi:.0%}"
        bucket_trades = [
            t for t in trades
            if t.event_prob_at_entry is not None
            and lo <= t.event_prob_at_entry < hi
            and t.pnl is not None
        ]
        if not bucket_trades:
            results[label] = {"count": 0, "avg_pnl": 0, "win_rate": 0, "total_pnl": 0}
            continue

        pnls = [t.pnl for t in bucket_trades]
        results[label] = {
            "count": len(bucket_trades),
            "avg_pnl": round(np.mean(pnls), 2),
            "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls), 2),
            "total_pnl": round(sum(pnls), 2),
        }

    return results


def signal_decay_analysis(
    trades: Sequence[BacktestTrade],
) -> pd.DataFrame:
    """
    Analyze how signal strength decays over time.

    Groups trades by entry month and computes rolling metrics.
    """
    completed = [t for t in trades if t.pnl is not None]
    if not completed:
        return pd.DataFrame()

    data = []
    for t in completed:
        data.append({
            "entry_date": t.entry_date,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct or 0.0,
            "event_prob": t.event_prob_at_entry or 0.0,
        })

    df = pd.DataFrame(data)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["rolling_win_rate"] = df["pnl"].apply(lambda x: 1 if x > 0 else 0).rolling(5, min_periods=1).mean()

    return df


def calibration_error(
    predicted_probs: Sequence[float],
    realized_outcomes: Sequence[bool],
    n_bins: int = 5,
) -> float:
    """
    Expected Calibration Error (ECE) for the model's probability estimates.

    Measures whether events the model says are 30% likely actually happen ~30% of the time.
    """
    if len(predicted_probs) != len(realized_outcomes) or len(predicted_probs) == 0:
        return 0.0

    probs = np.array(predicted_probs)
    outcomes = np.array(realized_outcomes, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
        if not np.any(mask):
            continue
        bin_prob = np.mean(probs[mask])
        bin_outcome = np.mean(outcomes[mask])
        bin_weight = np.sum(mask) / len(probs)
        ece += bin_weight * abs(bin_prob - bin_outcome)

    return float(ece)
