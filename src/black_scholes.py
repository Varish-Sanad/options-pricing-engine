import numpy as np
from scipy.stats import norm


# d1, d2 are z-scores locating the strike within the risk-neutral lognormal
# distribution of S_T. d2 uses the true risk-neutral drift (r - sigma^2/2), so
# N(d2) is a real probability of finishing ITM. d1 = d2 + sigma*sqrt(T) is the
# value-weighted version used for the stock term (this is also Delta).
def d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# Not re-derived independently - this is the call price formula mirrored via
# put-call parity (N(d) -> N(-d)), which is why it reuses the same d1/d2.
def put_price(S, K, T, r, sigma):
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
