import numpy as np


# Jumps straight to S_T using the closed-form lognormal solution to GBM
# (we don't need the intermediate path, only the terminal price, since these
# are European options). One Z per simulated "world."
def simulate_terminal_prices(S, T, r, sigma, num_simulations, seed=None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(num_simulations)
    return S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Z)


# Antithetic variates: only draw half as many independent Z's, then reuse
# each one mirrored (-Z) as a second, negatively-correlated sample. Cuts
# standard error roughly in half for the same number of draws.
def simulate_terminal_prices_antithetic(S, T, r, sigma, num_simulations, seed=None):
    rng = np.random.default_rng(seed)
    half = num_simulations // 2
    Z = rng.standard_normal(half)
    drift = (r - sigma**2 / 2) * T
    diffusion = sigma * np.sqrt(T)
    S_T_plus = S * np.exp(drift + diffusion * Z)
    S_T_minus = S * np.exp(drift - diffusion * Z)
    return S_T_plus, S_T_minus


def _price_from_payoffs(payoffs, r, T, num_paths):
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(num_paths)  # standard error of the mean
    return price, std_error


def mc_call_price(S, K, T, r, sigma, num_simulations=100_000, seed=None, antithetic=False):
    if antithetic:
        S_T_plus, S_T_minus = simulate_terminal_prices_antithetic(S, T, r, sigma, num_simulations, seed)
        # Average each (Z, -Z) pair into one payoff before computing std error -
        # that's what actually captures the variance reduction. Treating all
        # the raw payoffs as independent samples would understate it, since
        # the two halves of each pair aren't independent of each other.
        pair_payoffs = (np.maximum(S_T_plus - K, 0) + np.maximum(S_T_minus - K, 0)) / 2
        return _price_from_payoffs(pair_payoffs, r, T, len(pair_payoffs))
    S_T = simulate_terminal_prices(S, T, r, sigma, num_simulations, seed)
    payoffs = np.maximum(S_T - K, 0)
    return _price_from_payoffs(payoffs, r, T, num_simulations)


def mc_put_price(S, K, T, r, sigma, num_simulations=100_000, seed=None, antithetic=False):
    if antithetic:
        S_T_plus, S_T_minus = simulate_terminal_prices_antithetic(S, T, r, sigma, num_simulations, seed)
        pair_payoffs = (np.maximum(K - S_T_plus, 0) + np.maximum(K - S_T_minus, 0)) / 2
        return _price_from_payoffs(pair_payoffs, r, T, len(pair_payoffs))
    S_T = simulate_terminal_prices(S, T, r, sigma, num_simulations, seed)
    payoffs = np.maximum(K - S_T, 0)
    return _price_from_payoffs(payoffs, r, T, num_simulations)
