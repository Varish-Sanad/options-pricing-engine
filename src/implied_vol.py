import numpy as np

from black_scholes import call_price, put_price
from greeks import vega


# No-arbitrage bounds on price given S, K, T, r alone (independent of sigma).
# A quoted price outside this range implies no real sigma reproduces it.
def _price_bounds(S, K, T, r, option_type):
    discounted_K = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - discounted_K, 0.0), S
    return max(discounted_K - S, 0.0), discounted_K


# Bisection is slower than Newton but can't diverge - used only as a fallback
# when Newton's vega-based step misbehaves (near-zero vega, bad initial guess).
def _bisection(price_fn, market_price, S, K, T, r, lo, hi, tol, max_iterations):
    f_lo = price_fn(S, K, T, r, lo) - market_price
    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        f_mid = price_fn(S, K, T, r, mid) - market_price
        if abs(f_mid) < tol:
            return mid
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return mid


def implied_volatility(
    market_price, S, K, T, r, option_type="call",
    initial_guess=0.20, tol=1e-6, max_iterations=100,
):
    lo_bound, hi_bound = _price_bounds(S, K, T, r, option_type)
    if not (lo_bound < market_price < hi_bound):
        raise ValueError(
            f"market_price={market_price} violates no-arbitrage bounds "
            f"({lo_bound:.4f}, {hi_bound:.4f}) - no real sigma reproduces it"
        )

    price_fn = call_price if option_type == "call" else put_price

    sigma = initial_guess
    for _ in range(max_iterations):
        price = price_fn(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega(S, K, T, r, sigma)
        if v < 1e-8:
            break  # flat enough that Newton's step is unreliable - fall through to bisection

        sigma -= diff / v
        sigma = min(max(sigma, 1e-4), 5.0)  # keep the guess in a sane vol range

    # Fixed, generous budget - independent of whatever max_iterations the
    # caller gave Newton, so a low Newton cap can't starve the fallback too.
    return _bisection(price_fn, market_price, S, K, T, r, 1e-6, 5.0, tol, 200)
