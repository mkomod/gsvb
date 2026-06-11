"""Evidence Lower Bound (ELBO). Port of R ``gsvb.elbo``.

Note: unlike the R version (which passes the pre-intercept column count), this
uses the actual coefficient dimension so the residual term covers every
coefficient.
"""

import numpy as np

from . import _gsvb_core as _core


def gsvb_elbo(fit, y, X, mcn=500, approx=False, approx_thresh=1e-3):
    """Compute the ELBO of a fitted model at data ``(y, X)``."""
    params = fit["parameters"]
    y = np.ascontiguousarray(np.asarray(y, dtype=float).ravel())
    X = np.ascontiguousarray(np.asarray(X, dtype=float))
    groups = np.ascontiguousarray(np.asarray(params["groups"]).astype(np.uint64))
    grp_index = np.asarray(params["groups"]).astype(int)
    grp_index = grp_index - grp_index.min()

    if params["intercept"]:
        X = np.ascontiguousarray(np.column_stack([np.ones(X.shape[0]), X]))

    n, p = X.shape
    mu = np.asarray(fit["mu"], dtype=float)
    g_full = np.asarray(fit["g"], dtype=float)[grp_index]
    lam = params["lambda"]
    a0, b0 = params["a0"], params["b0"]
    fam = params["family"]
    diag = params["diag_covariance"]

    if fam == 1:
        yty = float(np.dot(y, y))
        yx = X.T @ y
        xtx = X.T @ X
        ta, tb = fit["tau_a"], fit["tau_b"]
        ta0, tb0 = params["tau_a0"], params["tau_b0"]
        if diag:
            return _core.elbo_linear_c(
                yty, yx, xtx, groups, n, p, mu, np.asarray(fit["s"], dtype=float),
                g_full, ta, tb, lam, a0, b0, ta0, tb0, int(mcn), approx,
                approx_thresh)
        Ss = [np.asarray(S, dtype=float) for S in fit["s"]]
        return _core.elbo_linear_u(
            yty, yx, xtx, groups, n, p, mu, Ss, g_full, ta, tb, lam, a0, b0,
            ta0, tb0, int(mcn), approx, approx_thresh)

    w = a0 / (a0 + b0)

    if fam in (2, 3, 4):
        if diag:
            s = np.asarray(fit["s"], dtype=float)
            Ss = [np.zeros((1, 1))]
        else:
            s = np.ones(p)
            Ss = [np.asarray(S, dtype=float) for S in fit["s"]]
        return _core.elbo_logistic(y, X, groups, mu, s, g_full, Ss, lam, w,
                                   int(mcn), diag)

    # poisson
    if diag:
        return _core.elbo_poisson(y, X, groups, mu, np.asarray(fit["s"], dtype=float),
                                  g_full, lam, w, int(mcn))
    Ss = [np.asarray(S, dtype=float) for S in fit["s"]]
    return _core.elbo_poisson_S(y, X, groups, mu, Ss, g_full, lam, w, int(mcn))
