from black_scholes import call_price, put_price
from greeks import delta, gamma, vega, theta, rho, delta_fd, gamma_fd, vega_fd, theta_fd, rho_fd
from monte_carlo import mc_call_price, mc_put_price

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20

print(f"Inputs: S={S} K={K} T={T} r={r} sigma={sigma}")

print("\n--- Black-Scholes ---")
bs_call = call_price(S, K, T, r, sigma)
bs_put = put_price(S, K, T, r, sigma)
print(f"call = {bs_call:.4f}")
print(f"put  = {bs_put:.4f}")

print("\n--- Greeks: analytical vs. finite-difference (call) ---")
greeks = [
    ("delta", delta(S, K, T, r, sigma, "call"), delta_fd(S, K, T, r, sigma, "call")),
    ("gamma", gamma(S, K, T, r, sigma), gamma_fd(S, K, T, r, sigma, "call")),
    ("vega", vega(S, K, T, r, sigma), vega_fd(S, K, T, r, sigma, "call")),
    ("theta", theta(S, K, T, r, sigma, "call"), theta_fd(S, K, T, r, sigma, "call")),
    ("rho", rho(S, K, T, r, sigma, "call"), rho_fd(S, K, T, r, sigma, "call")),
]
for name, analytic, fd in greeks:
    print(f"{name:6s} analytic={analytic:9.4f}  finite-diff={fd:9.4f}  diff={abs(analytic - fd):.2e}")

print("\n--- Monte Carlo convergence vs. Black-Scholes (call) ---")
for n in [1_000, 10_000, 100_000, 1_000_000]:
    plain, plain_se = mc_call_price(S, K, T, r, sigma, num_simulations=n, seed=42)
    anti, anti_se = mc_call_price(S, K, T, r, sigma, num_simulations=n, seed=42, antithetic=True)
    print(
        f"N={n:>9,}  plain={plain:.4f} +/- {plain_se:.4f}   "
        f"antithetic={anti:.4f} +/- {anti_se:.4f}   (BS={bs_call:.4f})"
    )
