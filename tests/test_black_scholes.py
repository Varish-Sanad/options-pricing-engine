import numpy as np
import pytest

from black_scholes import call_price, put_price

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20


def test_call_price_matches_textbook_example():
    # Hull, "Options, Futures, and Other Derivatives" — standard reference example
    assert call_price(S, K, T, r, sigma) == pytest.approx(4.76, abs=1e-2)


def test_put_price_matches_textbook_example():
    assert put_price(S, K, T, r, sigma) == pytest.approx(0.81, abs=1e-2)


def test_put_call_parity():
    c = call_price(S, K, T, r, sigma)
    p = put_price(S, K, T, r, sigma)
    assert c - p == pytest.approx(S - K * np.exp(-r * T), abs=1e-8)
