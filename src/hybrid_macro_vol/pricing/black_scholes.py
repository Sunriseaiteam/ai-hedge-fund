"""Black-Scholes option pricing and Greeks."""

from __future__ import annotations

import math

from scipy.stats import norm

from src.hybrid_macro_vol.models import OptionType


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 parameter of Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 parameter of Black-Scholes."""
    return d1(S, K, T, r, sigma) - sigma * math.sqrt(T)


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType = OptionType.PUT,
) -> float:
    """
    European option price under Black-Scholes.

    Args:
        S: Spot price of underlying
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (annualized, e.g. 0.05 for 5%)
        sigma: Volatility (annualized, e.g. 0.20 for 20%)
        option_type: CALL or PUT
    """
    if T <= 0:
        if option_type == OptionType.CALL:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    if sigma <= 0:
        df = math.exp(-r * T)
        if option_type == OptionType.CALL:
            return max(S - K * df, 0.0)
        return max(K * df - S, 0.0)

    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)

    if option_type == OptionType.CALL:
        return S * norm.cdf(_d1) - K * math.exp(-r * T) * norm.cdf(_d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-_d2) - S * norm.cdf(-_d1)


def bs_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = OptionType.PUT,
) -> float:
    """Option delta."""
    if T <= 0 or sigma <= 0:
        if option_type == OptionType.CALL:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    _d1 = d1(S, K, T, r, sigma)
    if option_type == OptionType.CALL:
        return norm.cdf(_d1)
    return norm.cdf(_d1) - 1.0


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option gamma (same for calls and puts)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return norm.pdf(_d1) / (S * sigma * math.sqrt(T))


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Option vega (per 1% vol move). Same for calls and puts."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    return S * norm.pdf(_d1) * math.sqrt(T) * 0.01


def bs_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    option_type: OptionType = OptionType.PUT,
) -> float:
    """Option theta (per calendar day)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma)
    _d2 = d2(S, K, T, r, sigma)
    common = -(S * norm.pdf(_d1) * sigma) / (2.0 * math.sqrt(T))
    if option_type == OptionType.CALL:
        return (common - r * K * math.exp(-r * T) * norm.cdf(_d2)) / 365.0
    return (common + r * K * math.exp(-r * T) * norm.cdf(-_d2)) / 365.0


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType = OptionType.PUT,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Compute implied volatility via Newton-Raphson.

    Returns None if convergence fails.
    """
    if T <= 0 or market_price <= 0:
        return None

    sigma = 0.25  # initial guess
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        v = bs_vega(S, K, T, r, sigma) * 100  # vega was per 1%, need per 100%
        if v < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / v
        sigma = max(sigma, 0.001)
        sigma = min(sigma, 5.0)

    return sigma if abs(bs_price(S, K, T, r, sigma, option_type) - market_price) < 0.01 else None
