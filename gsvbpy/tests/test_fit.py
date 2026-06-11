"""End-to-end functional tests for the fit/predict/sample/elbo wrappers."""

import numpy as np
import pytest

import gsvbpy


def _make_groups(p, gsize):
    return np.repeat(np.arange(1, p // gsize + 1), gsize)


def test_gaussian_recovers_signal():
    rng = np.random.default_rng(0)
    n, p, gsize = 120, 60, 5
    groups = _make_groups(p, gsize)
    X = rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = -4.0
    b[10:15] = 8.0
    y = X @ b + rng.standard_normal(n)

    gsvbpy.set_seed(1)
    f = gsvbpy.gsvb_fit(y, X, groups, family="gaussian", verbose=False, niter=100)

    # beta_hat excludes the intercept (first coefficient)
    beta = f["beta_hat"][1:]
    active = np.where(np.abs(b) > 0)[0]
    inactive = np.where(np.abs(b) == 0)[0]

    # active coefficients recovered with the right sign/scale
    assert np.corrcoef(beta[active], b[active])[0, 1] > 0.95
    # inactive coefficients shrunk towards zero
    assert np.abs(beta[inactive]).max() < 1.0
    assert f["tau_hat"] > 0


def test_predict_sample_credible_elbo():
    rng = np.random.default_rng(2)
    n, p, gsize = 80, 30, 5
    groups = _make_groups(p, gsize)
    X = rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = 5.0
    y = X @ b + rng.standard_normal(n)

    gsvbpy.set_seed(3)
    f = gsvbpy.gsvb_fit(y, X, groups, family="gaussian", verbose=False, niter=80)

    pred = gsvbpy.gsvb_predict(f, X, samples=500, seed=3)
    assert pred["mean"].shape == (n,)
    assert pred["quantiles"].shape == (2, n)

    s = gsvbpy.gsvb_sample(f, samples=300, seed=3)
    assert s["beta"].shape[1] == 300
    assert "tau" in s

    ci = gsvbpy.gsvb_credible_intervals(f, prob=0.95)
    assert ci["lower"].shape == f["mu"].shape
    assert np.all(ci["lower"] <= ci["upper"])

    e = gsvbpy.gsvb_elbo(f, y, X, mcn=200)
    assert np.isfinite(e)


def test_logistic_jaakkola_runs():
    rng = np.random.default_rng(4)
    n, p, gsize = 100, 30, 5
    groups = _make_groups(p, gsize)
    X = rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = 3.0
    prob = 1.0 / (1.0 + np.exp(-(X @ b)))
    y = (rng.random(n) < prob).astype(float)

    gsvbpy.set_seed(5)
    f = gsvbpy.gsvb_fit(y, X, groups, family="binomial-jaakkola",
                        verbose=False, niter=80)
    assert f["beta_hat"].shape[0] == p + 1          # + intercept
    # the active group should have the largest inclusion probability
    assert f["g"][2] > 0.5                            # group 2 -> coefficients 5:10


def test_poisson_runs():
    rng = np.random.default_rng(6)
    n, p, gsize = 100, 20, 5
    groups = _make_groups(p, gsize)
    X = 0.3 * rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = 0.8
    y = rng.poisson(np.exp(X @ b)).astype(float)

    gsvbpy.set_seed(7)
    f = gsvbpy.gsvb_fit(y, X, groups, family="poisson", verbose=False, niter=80)
    assert np.isfinite(f["beta_hat"]).all()


@pytest.mark.parametrize("family", ["binomial-jensens", "binomial-jaakkola",
                                    "binomial-refined"])
def test_binomial_bounds_identify_active_group(family):
    rng = np.random.default_rng(10)
    n, p, gsize = 120, 30, 5
    groups = _make_groups(p, gsize)
    X = rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = 3.0
    prob = 1.0 / (1.0 + np.exp(-(X @ b)))
    y = (rng.random(n) < prob).astype(float)

    gsvbpy.set_seed(11)
    f = gsvbpy.gsvb_fit(y, X, groups, family=family, verbose=False, niter=60)
    assert f["beta_hat"].shape[0] == p + 1
    assert np.isfinite(f["beta_hat"]).all()
    assert f["g"][2] > 0.5            # group 2 -> active coefficients 5:10


@pytest.mark.parametrize("family", ["gaussian", "binomial-jaakkola", "poisson"])
def test_full_covariance(family):
    rng = np.random.default_rng(12)
    n, p, gsize = 120, 30, 5
    groups = _make_groups(p, gsize)
    X = 0.3 * rng.standard_normal((n, p))
    b = np.zeros(p)
    b[5:10] = 1.0
    if family == "gaussian":
        y = X @ b + rng.standard_normal(n)
    elif family == "poisson":
        y = rng.poisson(np.exp(X @ b)).astype(float)
    else:
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(X @ b)))).astype(float)

    gsvbpy.set_seed(13)
    f = gsvbpy.gsvb_fit(y, X, groups, family=family, diag_covariance=False,
                        verbose=False, niter=60)
    # s is a list of per-group covariance matrices
    assert isinstance(f["s"], list)
    assert np.asarray(f["s"][2]).shape == (gsize, gsize)
    assert np.isfinite(f["beta_hat"]).all()
    ci = gsvbpy.gsvb_credible_intervals(f)
    assert ci["lower"].shape == f["mu"].shape


def test_jensen_refined_force_diagonal():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((40, 10))
    y = (rng.random(40) < 0.5).astype(float)
    groups = _make_groups(10, 5)
    with pytest.warns(UserWarning, match="diagonal"):
        f = gsvbpy.gsvb_fit(y, X, groups, family="binomial-jensens",
                            diag_covariance=False, verbose=False, niter=10)
    assert f["parameters"]["diag_covariance"] is True


def test_invalid_family_raises():
    rng = np.random.default_rng(8)
    X = rng.standard_normal((20, 10))
    y = rng.standard_normal(20)
    groups = _make_groups(10, 5)
    with pytest.raises(ValueError):
        gsvbpy.gsvb_fit(y, X, groups, family="not-a-family")
