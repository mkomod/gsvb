"""Sample from the variational posterior. Port of R ``gsvb.sample``."""

import numpy as np


def gsvb_sample(fit, samples=10000, seed=None):
    """Draw ``samples`` samples of beta (and tau, for the linear model).

    Returns a dict with ``beta`` of shape ``(p, samples)`` and, for the gaussian
    family, ``tau`` of length ``samples``.
    """
    rng = np.random.default_rng(seed)
    params = fit["parameters"]
    groups = np.asarray(params["groups"]).astype(int)
    g = np.asarray(fit["g"], dtype=float)
    mu = np.asarray(fit["mu"], dtype=float)
    p = mu.size
    M = g.size

    # group label per coefficient is 1..M; map to 0-based group index
    grp_index = groups - groups.min()

    beta = np.zeros((p, samples))
    diag = params["diag_covariance"]

    if diag:
        s = np.asarray(fit["s"], dtype=float)

    for k in range(samples):
        active = rng.random(M) <= g                 # per-group inclusion
        active_coef = active[grp_index]             # per-coefficient mask
        if not active_coef.any():
            continue
        if diag:
            idx = np.where(active_coef)[0]
            beta[idx, k] = rng.normal(mu[idx], s[idx])
        else:
            S_list = fit["s"]
            for j in np.where(active)[0]:
                Gj = np.where(grp_index == j)[0]
                L = np.linalg.cholesky(np.asarray(S_list[j], dtype=float))
                beta[Gj, k] = L @ rng.standard_normal(Gj.size) + mu[Gj]

    out = {"beta": beta}
    if params["family"] == 1:
        out["tau"] = 1.0 / rng.gamma(shape=fit["tau_a"], scale=1.0 / fit["tau_b"],
                                     size=samples)
    return out
