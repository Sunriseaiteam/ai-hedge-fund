"""Core data models for the hybrid macro-vol engine."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    AVOID = "avoid"


class SignalStrength(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    AVOID = "avoid"
    SELL = "sell"


# ──────────────────────────────────────────────
# Prediction Markets
# ──────────────────────────────────────────────

class PredictionMarketEvent(BaseModel):
    """A single prediction market contract."""
    source: str  # "polymarket" or "kalshi"
    event_id: str
    title: str
    probability: float = Field(ge=0.0, le=1.0)
    bid: float | None = None
    ask: float | None = None
    volume: float | None = None
    open_interest: float | None = None
    settlement_date: date | None = None
    category: str | None = None
    timestamp: datetime | None = None
    url: str | None = None


# ──────────────────────────────────────────────
# Options
# ──────────────────────────────────────────────

class OptionQuote(BaseModel):
    """A single option contract quote."""
    ticker: str
    expiry: date
    strike: float
    option_type: OptionType
    bid: float
    ask: float
    mid: float
    implied_vol: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    underlying_price: float | None = None


class OptionChain(BaseModel):
    """Full option chain for a ticker."""
    ticker: str
    underlying_price: float
    risk_free_rate: float
    as_of: datetime
    quotes: list[OptionQuote]


# ──────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────

class Scenario(BaseModel):
    """A single macro scenario with market impact parameters."""
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    spot_shock: float  # e.g. -0.25 means SPY drops 25%
    volatility: float  # annualized vol in that regime
    rate_shift: float = 0.0  # shift to risk-free rate (bps → decimal)
    description: str = ""


class ScenarioSet(BaseModel):
    """A named collection of scenarios for an event type."""
    name: str
    event_type: str  # e.g. "recession", "fed_cut", "crisis"
    scenarios: list[Scenario]
    source_description: str = ""

    def total_probability(self) -> float:
        return sum(s.probability for s in self.scenarios)

    def validate_probabilities(self) -> bool:
        return abs(self.total_probability() - 1.0) < 1e-6


# ──────────────────────────────────────────────
# Pricing Results
# ──────────────────────────────────────────────

class PricingResult(BaseModel):
    """Result of pricing a single option under the hybrid model."""
    ticker: str
    expiry: date
    strike: float
    option_type: OptionType
    underlying_price: float

    # Prices
    market_mid: float
    bs_value: float  # plain Black-Scholes at current vol
    hybrid_value: float  # scenario-weighted value

    # Edge metrics
    edge: float  # hybrid_value - market_mid
    edge_pct: float  # edge / market_mid
    edge_per_contract: float  # edge * 100

    # Break-even
    breakeven_probability: float | None = None  # prob needed to justify market price

    # Per-scenario breakdown
    scenario_values: dict[str, float] = {}  # scenario_name → option value

    # Context
    implied_vol: float | None = None
    days_to_expiry: int = 0
    bid: float = 0.0
    ask: float = 0.0


# ──────────────────────────────────────────────
# Signal Output
# ──────────────────────────────────────────────

class OptionSignal(BaseModel):
    """A scored trade signal for a single option."""
    pricing: PricingResult
    signal: SignalStrength
    edge_score: float  # normalized composite score
    liquidity_score: float = 0.0
    convexity_score: float = 0.0
    notes: str = ""


# ──────────────────────────────────────────────
# Trade Recommendations
# ──────────────────────────────────────────────

class TradeLeg(BaseModel):
    """A single leg in a trade structure."""
    action: TradeAction
    option_type: OptionType
    strike: float
    expiry: date
    quantity: int = 1
    price: float = 0.0


class TradeRecommendation(BaseModel):
    """A recommended trade structure."""
    name: str  # e.g. "Long Put", "Put Spread 550/500"
    legs: list[TradeLeg]
    max_loss: float | None = None
    max_gain: float | None = None
    breakeven: float | None = None
    net_debit: float | None = None
    edge_score: float = 0.0
    scenario_pnl: dict[str, float] = {}  # scenario → expected pnl
    rationale: str = ""


# ──────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────

class BacktestTrade(BaseModel):
    """A single historical trade in a backtest."""
    entry_date: date
    exit_date: date | None = None
    ticker: str
    structure: str
    entry_price: float
    exit_price: float | None = None
    pnl: float | None = None
    pnl_pct: float | None = None
    scenario_at_entry: str = ""
    event_prob_at_entry: float | None = None


class BacktestResult(BaseModel):
    """Aggregate backtest results."""
    trades: list[BacktestTrade]
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float | None = None
    by_regime: dict[str, float] = {}
    by_event_type: dict[str, float] = {}
