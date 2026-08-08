import pytest

from black_scholes import call_price, put_price
from monte_carlo import mc_call_price, mc_put_price

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20
N = 200_000


def test_mc_call_converges_to_black_scholes():
    bs = call_price(S, K, T, r, sigma)
    mc, se = mc_call_price(S, K, T, r, sigma, num_simulations=N, seed=42)
    assert mc == pytest.approx(bs, abs=4 * se)


def test_mc_put_converges_to_black_scholes():
    bs = put_price(S, K, T, r, sigma)
    mc, se = mc_put_price(S, K, T, r, sigma, num_simulations=N, seed=42)
    assert mc == pytest.approx(bs, abs=4 * se)


def test_antithetic_reduces_standard_error():
    _, plain_se = mc_call_price(S, K, T, r, sigma, num_simulations=N, seed=42)
    _, anti_se = mc_call_price(S, K, T, r, sigma, num_simulations=N, seed=42, antithetic=True)
    assert anti_se < plain_se
