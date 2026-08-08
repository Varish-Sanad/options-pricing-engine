import pytest

from greeks import (
    delta, gamma, vega, theta, rho,
    delta_fd, gamma_fd, vega_fd, theta_fd, rho_fd,
)

S, K, T, r, sigma = 42, 40, 0.5, 0.10, 0.20


def test_call_put_delta_parity():
    assert delta(S, K, T, r, sigma, "call") - delta(S, K, T, r, sigma, "put") == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize(
    "analytic_fn, fd_fn",
    [
        (delta, delta_fd),
        (theta, theta_fd),
        (rho, rho_fd),
    ],
)
def test_analytical_matches_finite_difference(analytic_fn, fd_fn, option_type):
    analytic = analytic_fn(S, K, T, r, sigma, option_type)
    fd = fd_fn(S, K, T, r, sigma, option_type)
    assert analytic == pytest.approx(fd, abs=1e-4)


def test_gamma_matches_finite_difference():
    assert gamma(S, K, T, r, sigma) == pytest.approx(gamma_fd(S, K, T, r, sigma), abs=1e-4)


def test_vega_matches_finite_difference():
    assert vega(S, K, T, r, sigma) == pytest.approx(vega_fd(S, K, T, r, sigma), abs=1e-4)
