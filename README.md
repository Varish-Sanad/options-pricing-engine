# Options Pricing Engine

An options pricing engine in Python implementing closed-form Black-Scholes pricing for European calls/puts, a Monte Carlo pricer using geometric Brownian motion path simulation, full Greeks (Delta, Gamma, Vega, Theta, Rho) via both analytical derivatives and finite-difference approximation, and antithetic-variate variance reduction for the Monte Carlo engine.

## Why I built this

This project is my entry point into quantitative finance. I came in with a physics/linear-algebra background and no prior exposure to stochastic calculus or derivatives pricing, and used this as a hands-on way to actually learn the field rather than just read about it — working through *why* Black-Scholes has the form it does (the no-arbitrage hedging argument, risk-neutral pricing), what each Greek actually measures and why a trading desk cares about it, and why Monte Carlo methods need variance reduction at all, before writing any code.

Every piece here is independently verified rather than just implemented once and trusted:
- Monte Carlo is cross-validated against the closed-form Black-Scholes price.
- Every analytical Greek is cross-validated against a finite-difference approximation of the same quantity, computed a completely independent way.
- Pricing is validated against a standard textbook reference example (Hull, *Options, Futures, and Other Derivatives*).

That verification approach is deliberate — it's the same instinct I'd want to bring to any quant role: don't just trust that a model runs, prove it's producing the right numbers.

## What I learned

- **The no-arbitrage / hedging argument** behind Black-Scholes: why a portfolio of an option plus the right number of shares can be made instantaneously riskless, and why that forces the portfolio to grow at exactly the risk-free rate.
- **Risk-neutral pricing**: why option prices don't depend on anyone's real-world view of where the stock is headed, only on hedgeable, observable quantities (r, σ).
- **What d1 and d2 actually represent**: standardized (z-score) distances within the lognormal distribution of the future stock price, not arbitrary formula artifacts.
- **What each Greek measures and why it's traded**: Delta as hedge ratio, Gamma as hedge-staleness risk, Vega as volatility exposure, Theta as time decay, Rho as rate exposure.
- **Why Monte Carlo needs variance reduction**: the `1/√N` convergence rate makes brute-force simulation slow, and antithetic variates (pairing every random draw with its mirror) cuts the standard error roughly in half at no extra computational cost.

## Results

Validated against Hull's standard example (S=42, K=40, T=0.5, r=10%, σ=20%):

| Method | Call | Put |
|---|---|---|
| Black-Scholes (closed-form) | 4.7594 | 0.8086 |
| Monte Carlo (1,000,000 paths, antithetic) | 4.7612 ± 0.0025 | — |

All 5 Greeks (Delta, Gamma, Vega, Theta, Rho) match their finite-difference approximations to within ~1e-7. Antithetic variates consistently reduce Monte Carlo standard error by ~50% versus plain simulation at the same sample size.

Full test suite: 15 tests across pricing, Greeks, and Monte Carlo convergence — see [`tests/`](tests/).

## Usage

```bash
pip install -r requirements.txt
python src/main.py    # runs pricing, Greeks, and MC convergence demo
pytest                 # runs the full verification suite
```

## Roadmap

- [x] Black-Scholes closed-form pricer (European call/put)
- [x] Monte Carlo pricer (GBM path simulation)
- [x] Convergence cross-validation between the two methods
- [x] Greeks: analytical derivatives
- [x] Greeks: finite-difference verification
- [x] Antithetic variates for variance reduction

## Tech Stack

Python, NumPy, SciPy, pytest
