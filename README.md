# Options Pricing Engine

Python implementation of an options pricing engine — closed-form Black-Scholes for European calls/puts, a Monte Carlo pricer built on GBM path simulation, all five main Greeks (Delta, Gamma, Vega, Theta, Rho) computed both analytically and via finite-difference, and antithetic variates to cut down Monte Carlo variance.

## Background

I came into this with a physics/linear algebra background but no prior exposure to stochastic calculus or derivatives pricing. Built this project as a way to actually learn the material by implementing it, rather than just reading about it — worked through why Black-Scholes takes the form it does (the no-arbitrage hedging argument, risk-neutral pricing), what each Greek is actually measuring, and why Monte Carlo needs variance reduction at all, before writing any code.

I also didn't want to just trust a single implementation, so everything here is checked two independent ways: Monte Carlo against the closed-form price, every analytical Greek against its own finite-difference approximation, and the whole pricer against a standard textbook example (Hull, *Options, Futures, and Other Derivatives*).

## What I actually learned building this

- The no-arbitrage/hedging argument — why an option plus the right number of shares can be made instantaneously riskless, and why that forces the combined position to grow at exactly the risk-free rate.
- Risk-neutral pricing: the price doesn't depend on anyone's opinion of where the stock is headed, only on r and σ.
- What d1 and d2 actually are — z-scores locating the strike inside the lognormal distribution of the future stock price, not just formula noise.
- Delta = hedge ratio, Gamma = how fast that hedge goes stale, Vega = exposure to volatility expectations, Theta = time decay, Rho = rate exposure.
- Why plain Monte Carlo converges so slowly (error shrinks with 1/√N) and how antithetic variates — pairing every random draw with its mirror — roughly halves the standard error for free.

## Results

Checked against Hull's textbook example (S=42, K=40, T=0.5, r=10%, σ=20%):

- Black-Scholes: call = 4.7594, put = 0.8086
- Monte Carlo, 1,000,000 paths, antithetic: call = 4.7612 ± 0.0025

All five Greeks match their finite-difference versions to about 1e-7. In every test I ran, antithetic variates cut the Monte Carlo standard error roughly in half versus plain simulation at the same sample size.

15 tests in `tests/` cover pricing, the Greeks, and Monte Carlo convergence.

## Running it

```bash
pip install -r requirements.txt
python src/main.py    # pricing, Greeks, and MC convergence demo
pytest                 # runs the test suite
```

## Status

- [x] Black-Scholes (European call/put)
- [x] Monte Carlo (GBM path simulation)
- [x] Monte Carlo vs. Black-Scholes cross-validation
- [x] Analytical Greeks
- [x] Finite-difference verification
- [x] Antithetic variates

## Stack

Python, NumPy, SciPy, pytest
