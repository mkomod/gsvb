"""Fit high-dimensional group-sparse regression models.

Python port of the GSVB R package's ``gsvb.fit``. The variational inference
itself runs in the compiled C++ backend (Armadillo + Ensmallen); this module
handles input checking, intercept/initialization bookkeeping and dispatch.
"""

import numpy as np

from . import _gsvb_core as _core

# family name -> integer code (matching the R package's internal coding)
_FAMILIES = {
    "gaussian": 1,
    "binomial-jensens": 2,
    "binomial-jaakkola": 3,
    "binomial-refined": 4,
    "poisson": 5,
}

# families wired up in this (core-first) port
_SUPPORTED = {"gaussian", "binomial-jaakkola", "poisson"}

_INIT_METHODS = {"ridge": 3, "random": 2, "lasso": 1}


def _initialize_mu(X, y, family, init_method, rng):
    """Starting value for ``mu`` via scikit-learn (init only affects convergence)."""
    if init_method == "random":
        return rng.normal(0.0, 0.5, size=X.shape[1])

    from sklearn.linear_model import (
        Lasso, Ridge, LogisticRegression, PoissonRegressor,
    )

    if family == 1:  # gaussian
        if init_method == "lasso":
            est = Lasso(alpha=1e-2, fit_intercept=False, max_iter=10000)
        else:  # ridge
            est = Ridge(alpha=1e-3, fit_intercept=False)
        est.fit(X, y)
        return np.asarray(est.coef_, dtype=float).ravel()

    if family in (2, 3, 4):  # binomial
        if init_method == "lasso":
            est = LogisticRegression(
                penalty="l1", C=10.0, fit_intercept=False,
                solver="liblinear", max_iter=10000,
            )
        else:  # ridge (default L2)
            est = LogisticRegression(C=10.0, fit_intercept=False, max_iter=10000)
        est.fit(X, (y > 0.5).astype(int))
        return np.asarray(est.coef_, dtype=float).ravel()

    # poisson
    est = PoissonRegressor(alpha=1e-3, fit_intercept=False, max_iter=10000)
    est.fit(X, y)
    return np.asarray(est.coef_, dtype=float).ravel()


def gsvb_fit(
    y, X, groups, family="gaussian", intercept=True, diag_covariance=True,
    lambda_=1.0, a0=1.0, b0=None, tau_a0=1e-3, tau_b0=1e-3,
    mu=None, s=None, g=None, track_elbo=True, track_elbo_every=5,
    track_elbo_mcn=500, niter=150, tol=1e-3, verbose=False,
    thresh=0.02, l=5, ordering=2, init_method="ridge", seed=None,
):
    """Fit a group-sparse variational Bayes regression model.

    Parameters mirror the R ``gsvb.fit``. ``family`` is one of ``"gaussian"``,
    ``"binomial-jaakkola"`` or ``"poisson"`` (core-first port; the remaining
    binomial bounds are compiled but not yet exposed). Returns a dict with keys
    ``mu, s, g, beta_hat, parameters, converged, iter`` (plus ``tau_a/tau_b/
    tau_hat/elbo`` where applicable).
    """
    if family not in _FAMILIES:
        raise ValueError(f"Invalid family: {family!r}")
    if family not in _SUPPORTED:
        raise NotImplementedError(
            f"family {family!r} is compiled but not yet exposed in this port; "
            f"supported: {sorted(_SUPPORTED)}"
        )
    fam = _FAMILIES[family]

    if init_method is None:
        init_method = "ridge"
    if init_method not in _INIT_METHODS:
        raise ValueError(f"Invalid init_method: {init_method!r}")

    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    if X.ndim != 2:
        raise ValueError("X must be a 2-D matrix")
    groups = np.asarray(groups).ravel()

    if b0 is None:
        b0 = float(np.unique(groups).size)

    # ---- input checks (mirroring the R version) ----
    if groups.min() != 1:
        raise ValueError("group labels must start at 1")
    if groups.max() != np.unique(groups).size:
        raise ValueError("group labels must not exceed the unique number of groups")
    if not np.array_equal(groups, np.sort(groups)):
        raise ValueError("groups must be ordered")
    if any(v <= 0 for v in (lambda_, a0, b0, tau_a0, tau_b0)):
        raise ValueError("Hyperparameters must be greater than 0")
    if fam in (2, 3, 4) and not np.all((y == 0) | (y == 1)):
        raise ValueError("Classification requires y to be in {0, 1}")

    rng = np.random.default_rng(seed)
    if seed is not None:
        _core.set_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

    # non-gaussian families use a fixed tau and a different default for s
    if fam in (2, 3, 4, 5):
        tau_a0 = tau_b0 = 1.0
        if s is None:
            s = 1.0 / np.sqrt((X ** 2).sum(axis=0) + 2 * lambda_)

    if s is None:
        s = 1.0 / np.sqrt((X ** 2).sum(axis=0) * tau_a0 / tau_b0 + 2 * lambda_)
    s = np.asarray(s, dtype=float).ravel()

    if g is None:
        g = np.full(X.shape[1], 0.5)
    g = np.asarray(g, dtype=float).ravel()

    # ---- intercept: prepend a unit column and a singleton group ----
    if intercept:
        groups = np.concatenate(([groups.min() - 1], groups)) + 1
        X = np.ascontiguousarray(np.column_stack([np.ones(X.shape[0]), X]))
        n = X.shape[0]
        if s.size == X.shape[1] - 1:
            s0 = 1.0 / np.sqrt(np.sqrt(n) * tau_a0 / tau_b0 + 2 * lambda_)
            s = np.concatenate(([s0], s))
        if g.size != X.shape[1]:
            g = np.concatenate(([0.5], g))

    p = X.shape[1]

    # ---- initialize mu ----
    if mu is None:
        mu = _initialize_mu(X, y, fam, init_method, rng)
    mu = np.asarray(mu, dtype=float).ravel()

    groups_u = np.ascontiguousarray(groups.astype(np.uint64))

    # ---- dispatch ----
    if fam == 1:  # linear
        f = _core.fit_linear(
            y, X, groups_u, lambda_, a0, b0, tau_a0, tau_b0, mu, s, g,
            diag_covariance, track_elbo, int(track_elbo_every),
            int(track_elbo_mcn), int(niter), tol, verbose, int(ordering),
        )
    elif fam == 3:  # logistic, Jaakkola bound (alg=3)
        f = _core.fit_logistic(
            y, X, groups_u, lambda_, a0, b0, mu, s, g, diag_covariance,
            track_elbo, int(track_elbo_every), int(track_elbo_mcn), thresh,
            int(l), int(niter), 3, tol, verbose, int(ordering),
        )
    else:  # poisson
        f = _core.fit_poisson(
            y, X, groups_u, lambda_, a0, b0, mu, s, g, diag_covariance,
            track_elbo, int(track_elbo_every), int(track_elbo_mcn), int(niter),
            tol, verbose,
        )

    gamma = np.asarray(f["gamma"], dtype=float)
    mu_hat = np.asarray(f["mu"], dtype=float)

    # per-group inclusion probabilities (gamma at the first index of each group)
    _, first_idx = np.unique(groups, return_index=True)
    g_group = gamma[np.sort(first_idx)]

    # variational variance: std devs (diag) or list of covariance matrices
    if diag_covariance:
        s_out = np.asarray(f["sigma"], dtype=float)
    else:
        s_out = [np.asarray(S, dtype=float) for S in f["S"]]

    res = {
        "mu": mu_hat,
        "s": s_out,
        "g": g_group,
        "beta_hat": mu_hat * gamma,
        "parameters": {
            "lambda": lambda_, "a0": a0, "b0": b0,
            "intercept": intercept, "diag_covariance": diag_covariance,
            "groups": groups.astype(int), "family": fam,
        },
        "converged": bool(f["converged"]),
        "iter": int(f["iterations"]),
    }

    if fam == 1:
        res["tau_a"] = f["tau_a"]
        res["tau_b"] = f["tau_b"]
        res["tau_hat"] = f["tau_b"] / (f["tau_a"] - 1)
        res["parameters"]["tau_a0"] = tau_a0
        res["parameters"]["tau_b0"] = tau_b0

    if track_elbo:
        res["elbo"] = np.asarray(f["elbo"], dtype=float)

    return res
