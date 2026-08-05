# Options Pricing Engine

An options pricing engine in Python implementing closed-form Black-Scholes pricing for European calls/puts alongside a Monte Carlo pricer using geometric Brownian motion path simulation, with full Greeks calculation (Delta, Gamma, Vega, Theta, Rho) via analytical derivatives and finite-difference approximation, plus antithetic-variate variance reduction for the Monte Carlo engine.

## Status

In progress.

## Roadmap

- [ ] Black-Scholes closed-form pricer (European call/put)
- [ ] Monte Carlo pricer (GBM path simulation)
- [ ] Convergence cross-validation between the two methods
- [ ] Greeks: analytical derivatives
- [ ] Greeks: finite-difference verification
- [ ] Antithetic variates for variance reduction

## Tech Stack

Python, NumPy
