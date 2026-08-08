import numpy as np
from scipy.stats import norm

from black_scholes import call_price, d1_d2, put_price


# Delta = hedge ratio: shares of stock needed per option to be instantaneously
# riskless. N(d1) for a call; shifted by -1 for a put (put-call parity again).
def delta(S, K, T, r, sigma, option_type="call"):
    d1, _ = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


# How fast Delta drifts as S moves - i.e. how stale a hedge gets between
# rebalances. Same for calls and puts at the same strike.
def gamma(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# Sensitivity to sigma. Same for calls and puts.
def vega(S, K, T, r, sigma):
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


# Sign convention: this is -dC/dT, i.e. value lost as calendar time passes
# (T counts down toward expiry), not the derivative w.r.t. T itself.
def theta(S, K, T, r, sigma, option_type="call"):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    decay = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        return decay - r * K * np.exp(-r * T) * norm.cdf(d2)
    return decay + r * K * np.exp(-r * T) * norm.cdf(-d2)


def rho(S, K, T, r, sigma, option_type="call"):
    _, d2 = d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


# Everything below re-derives the Greeks numerically (central difference on
# the raw pricing formula, no calculus) as an independent check on the
# analytical formulas above. Should match to within a few h^2 of error.


def delta_fd(S, K, T, r, sigma, option_type="call", h=1e-2):
    price = call_price if option_type == "call" else put_price
    return (price(S + h, K, T, r, sigma) - price(S - h, K, T, r, sigma)) / (2 * h)


# Gamma is a second derivative, so it needs the 3-point stencil instead of
# the plain 2-point central difference used everywhere else here.
def gamma_fd(S, K, T, r, sigma, option_type="call", h=1e-2):
    price = call_price if option_type == "call" else put_price
    up = price(S + h, K, T, r, sigma)
    mid = price(S, K, T, r, sigma)
    down = price(S - h, K, T, r, sigma)
    return (up - 2 * mid + down) / h**2


def vega_fd(S, K, T, r, sigma, option_type="call", h=1e-4):
    price = call_price if option_type == "call" else put_price
    return (price(S, K, T, r, sigma + h) - price(S, K, T, r, sigma - h)) / (2 * h)


def theta_fd(S, K, T, r, sigma, option_type="call", h=1e-4):
    price = call_price if option_type == "call" else put_price
    return -(price(S, K, T + h, r, sigma) - price(S, K, T - h, r, sigma)) / (2 * h)


def rho_fd(S, K, T, r, sigma, option_type="call", h=1e-4):
    price = call_price if option_type == "call" else put_price
    return (price(S, K, T, r + h, sigma) - price(S, K, T, r - h, sigma)) / (2 * h)
