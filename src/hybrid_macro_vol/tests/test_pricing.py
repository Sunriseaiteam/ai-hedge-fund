"""Tests for the pricing engine."""

import math
from datetime import date, timedelta

import pytest

from src.hybrid_macro_vol.models import OptionQuote, OptionType, Scenario, ScenarioSet
from src.hybrid_macro_vol.pricing.black_scholes import (
    bs_price,
    bs_delta,
    bs_gamma,
    bs_vega,
    bs_theta,
    implied_vol,
)
from src.hybrid_macro_vol.pricing.hybrid_pricer import (
    hybrid_price,
    price_option_under_scenario,
    compute_breakeven_probability,
    price_single_option,
    scan_chain,
)
from src.hybrid_macro_vol.pricing.scenarios import (
    load_scenario_maps,
    blend_with_prediction_market,
)


# ─── Black-Scholes tests ─────────────────────────

class TestBlackScholes:
    """Test BS pricing and Greeks."""

    def test_put_call_parity(self):
        """Put-call parity: C - P = S - K*exp(-rT)."""
        S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
        call = bs_price(S, K, T, r, sigma, OptionType.CALL)
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        parity = call - put - (S - K * math.exp(-r * T))
        assert abs(parity) < 1e-10

    def test_atm_prices_positive(self):
        """ATM options should have positive value."""
        S, K, T, r, sigma = 540.0, 540.0, 0.25, 0.045, 0.18
        call = bs_price(S, K, T, r, sigma, OptionType.CALL)
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        assert call > 0
        assert put > 0

    def test_deep_itm_put_near_intrinsic(self):
        """Deep ITM put should be close to intrinsic."""
        S, K, T, r, sigma = 400.0, 540.0, 0.1, 0.045, 0.18
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        intrinsic = K * math.exp(-r * T) - S
        assert put > intrinsic * 0.95

    def test_deep_otm_put_small(self):
        """Deep OTM put should be very cheap."""
        S, K, T, r, sigma = 540.0, 300.0, 0.25, 0.045, 0.18
        put = bs_price(S, K, T, r, sigma, OptionType.PUT)
        assert put < 0.10

    def test_higher_vol_higher_price(self):
        """Higher vol should always increase option price."""
        S, K, T, r = 540.0, 500.0, 0.5, 0.045
        p1 = bs_price(S, K, T, r, 0.15, OptionType.PUT)
        p2 = bs_price(S, K, T, r, 0.30, OptionType.PUT)
        assert p2 > p1

    def test_expired_option(self):
        """Expired option = intrinsic value."""
        S, K = 540.0, 500.0
        put = bs_price(S, K, 0.0, 0.045, 0.20, OptionType.PUT)
        assert put == 0.0  # OTM put

        itm_put = bs_price(400.0, 500.0, 0.0, 0.045, 0.20, OptionType.PUT)
        assert itm_put == 100.0

    def test_delta_bounds(self):
        """Delta should be in [-1, 0] for puts, [0, 1] for calls."""
        S, K, T, r, sigma = 540.0, 540.0, 0.5, 0.045, 0.20
        call_delta = bs_delta(S, K, T, r, sigma, OptionType.CALL)
        put_delta = bs_delta(S, K, T, r, sigma, OptionType.PUT)
        assert 0 < call_delta < 1
        assert -1 < put_delta < 0
        assert abs(call_delta - put_delta - 1.0) < 1e-10

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        gamma = bs_gamma(540.0, 540.0, 0.5, 0.045, 0.20)
        assert gamma > 0

    def test_vega_positive(self):
        """Vega should always be positive."""
        vega = bs_vega(540.0, 540.0, 0.5, 0.045, 0.20)
        assert vega > 0

    def test_theta_negative_for_long(self):
        """Theta should be negative for long options (time decay)."""
        theta_put = bs_theta(540.0, 540.0, 0.5, 0.045, 0.20, OptionType.PUT)
        theta_call = bs_theta(540.0, 540.0, 0.5, 0.045, 0.20, OptionType.CALL)
        assert theta_put < 0
        assert theta_call < 0

    def test_implied_vol_roundtrip(self):
        """Computing IV from a BS price should recover the original vol."""
        S, K, T, r, sigma = 540.0, 520.0, 0.5, 0.045, 0.22
        price = bs_price(S, K, T, r, sigma, OptionType.PUT)
        recovered = implied_vol(price, S, K, T, r, OptionType.PUT)
        assert recovered is not None
        assert abs(recovered - sigma) < 0.001


# ─── Scenario engine tests ───────────────────────

class TestScenarios:
    """Test scenario loading and blending."""

    def test_load_scenario_maps(self):
        """Should load all scenario maps from YAML."""
        maps = load_scenario_maps()
        assert "recession" in maps
        assert "fed_rate_cut" in maps
        assert "tariff_escalation" in maps
        assert "government_shutdown" in maps

    def test_scenario_probabilities_sum_to_one(self):
        """All scenario sets should have probabilities summing to 1."""
        maps = load_scenario_maps()
        for name, ss in maps.items():
            assert ss.validate_probabilities(), f"{name} probabilities don't sum to 1"

    def test_blend_preserves_total_probability(self):
        """After blending, total probability should still be 1.0."""
        maps = load_scenario_maps()
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            adjusted = blend_with_prediction_market(maps["recession"], prob)
            assert abs(adjusted.total_probability() - 1.0) < 1e-6

    def test_blend_increases_event_probability(self):
        """Higher event probability should increase event scenario weights."""
        maps = load_scenario_maps()
        low = blend_with_prediction_market(maps["recession"], 0.10)
        high = blend_with_prediction_market(maps["recession"], 0.60)

        # Event scenarios (mild/severe/crisis) should have higher prob in 'high'
        low_event = sum(s.probability for s in low.scenarios[1:])
        high_event = sum(s.probability for s in high.scenarios[1:])
        assert high_event > low_event

    def test_blend_extreme_probabilities(self):
        """Edge cases: 0% and 100% event probability."""
        maps = load_scenario_maps()
        zero = blend_with_prediction_market(maps["recession"], 0.0)
        assert zero.scenarios[0].probability == pytest.approx(1.0, abs=1e-6)

        one = blend_with_prediction_market(maps["recession"], 1.0)
        assert one.scenarios[0].probability == pytest.approx(0.0, abs=1e-6)


# ─── Hybrid pricer tests ─────────────────────────

class TestHybridPricer:
    """Test the scenario-weighted pricing engine."""

    def _make_scenarios(self) -> ScenarioSet:
        return ScenarioSet(
            name="test",
            event_type="test",
            scenarios=[
                Scenario(name="normal", probability=0.7, spot_shock=0.0, volatility=0.16),
                Scenario(name="recession", probability=0.2, spot_shock=-0.15, volatility=0.28),
                Scenario(name="crisis", probability=0.1, spot_shock=-0.35, volatility=0.50),
            ],
        )

    def test_hybrid_price_is_weighted_average(self):
        """Hybrid price should be the probability-weighted average of scenario values."""
        scenarios = self._make_scenarios()
        S, K, T, r = 540.0, 500.0, 0.5, 0.045

        hybrid_val, breakdown = hybrid_price(S, K, T, r, scenarios.scenarios, OptionType.PUT)

        # Manual weighted sum
        expected = sum(
            s.probability * breakdown[s.name]
            for s in scenarios.scenarios
        )
        assert abs(hybrid_val - expected) < 1e-10

    def test_hybrid_exceeds_bs_for_otm_puts_with_tail_risk(self):
        """With tail-risk scenarios, hybrid value for OTM puts should exceed plain BS."""
        scenarios = self._make_scenarios()
        S, K, T, r = 540.0, 450.0, 0.5, 0.045

        hybrid_val, _ = hybrid_price(S, K, T, r, scenarios.scenarios, OptionType.PUT)
        bs_val = bs_price(S, K, T, r, 0.18, OptionType.PUT)

        # Hybrid should be higher because it accounts for crash scenarios
        assert hybrid_val > bs_val

    def test_crisis_scenario_dominates_deep_otm_put_value(self):
        """For deep OTM puts, the crisis scenario should contribute most of the value."""
        scenarios = self._make_scenarios()
        S, K, T, r = 540.0, 380.0, 0.5, 0.045

        _, breakdown = hybrid_price(S, K, T, r, scenarios.scenarios, OptionType.PUT)

        # Crisis scenario value should be much larger than normal
        assert breakdown["crisis"] > breakdown["normal"] * 10

    def test_breakeven_probability_reasonable(self):
        """Break-even probability should be between 0 and 1."""
        scenarios = self._make_scenarios()
        S, K, T, r = 540.0, 500.0, 0.5, 0.045
        market_price = bs_price(S, K, T, r, 0.20, OptionType.PUT)

        be = compute_breakeven_probability(S, K, T, r, market_price, scenarios, OptionType.PUT)
        assert be is not None
        assert 0.0 <= be <= 1.0

    def test_price_single_option(self):
        """Full pipeline for a single option should produce valid results."""
        scenarios = self._make_scenarios()
        quote = OptionQuote(
            ticker="SPY",
            expiry=date.today() + timedelta(days=90),
            strike=500.0,
            option_type=OptionType.PUT,
            bid=8.50,
            ask=9.00,
            mid=8.75,
            implied_vol=0.20,
        )

        result = price_single_option(quote, 540.0, 0.045, scenarios)
        assert result.ticker == "SPY"
        assert result.market_mid == 8.75
        assert result.bs_value > 0
        assert result.hybrid_value > 0
        assert result.edge == pytest.approx(result.hybrid_value - result.market_mid, abs=1e-10)

    def test_scan_chain_returns_sorted_results(self):
        """Scan should return results sorted by edge descending."""
        scenarios = self._make_scenarios()
        quotes = []
        for strike in [480, 500, 520]:
            quotes.append(OptionQuote(
                ticker="SPY",
                expiry=date.today() + timedelta(days=60),
                strike=float(strike),
                option_type=OptionType.PUT,
                bid=5.0,
                ask=5.50,
                mid=5.25,
                implied_vol=0.20,
            ))

        results = scan_chain(quotes, 540.0, 0.045, scenarios)
        assert len(results) > 0
        # Should be sorted by edge descending
        for i in range(len(results) - 1):
            assert results[i].edge >= results[i + 1].edge


# ─── Signal scoring tests ────────────────────────

class TestSignalScoring:
    """Test edge scoring and signal generation."""

    def test_generate_signal(self):
        """Should produce a valid signal from a pricing result."""
        from src.hybrid_macro_vol.signals.edge_scoring import generate_signal
        from src.hybrid_macro_vol.models import PricingResult

        result = PricingResult(
            ticker="SPY",
            expiry=date.today() + timedelta(days=60),
            strike=500.0,
            option_type=OptionType.PUT,
            underlying_price=540.0,
            market_mid=8.00,
            bs_value=6.50,
            hybrid_value=10.00,
            edge=2.00,
            edge_pct=0.25,
            edge_per_contract=200.0,
            breakeven_probability=0.22,
            scenario_values={"normal": 2.0, "recession": 15.0, "crisis": 45.0},
            days_to_expiry=60,
            bid=7.50,
            ask=8.50,
        )

        signal = generate_signal(result)
        assert signal.edge_score > 0
        assert signal.liquidity_score > 0
        assert signal.convexity_score > 0
        assert signal.pricing == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
