import pytest

from black_scholes import call_price, put_price
from implied_vol import implied_volatility

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_recovers_known_sigma_from_its_own_price(option_type):
    price_fn = call_price if option_type == "call" else put_price
    market_price = price_fn(S, K, T, r, sigma)
    recovered = implied_volatility(market_price, S, K, T, r, option_type)
    assert recovered == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("true_sigma", [0.05, 0.20, 0.80, 2.0])
def test_recovers_across_a_wide_vol_range(true_sigma):
    market_price = call_price(S, K, T, r, true_sigma)
    recovered = implied_volatility(market_price, S, K, T, r, "call")
    assert recovered == pytest.approx(true_sigma, abs=1e-5)


def test_rejects_price_below_arbitrage_bound():
    with pytest.raises(ValueError):
        implied_volatility(-1.0, S, K, T, r, "call")


def test_rejects_price_above_arbitrage_bound():
    with pytest.raises(ValueError):
        implied_volatility(S + 1.0, S, K, T, r, "call")


def test_bisection_fallback_still_converges_when_newton_is_cut_short():
    # Forces Newton to bail after a single step (nowhere near tol), so the
    # returned answer can only have come from the bisection fallback path.
    market_price = call_price(S, K, T, r, sigma)
    recovered = implied_volatility(market_price, S, K, T, r, "call", max_iterations=1)
    assert recovered == pytest.approx(sigma, abs=1e-4)
