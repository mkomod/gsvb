"""Marginal credible intervals. Port of R ``gsvb.credible_intervals``."""

import numpy as np
from scipy.stats import norm


def gsvb_credible_intervals(fit, prob=0.95):
    """Highest-posterior-density marginal credible interval per coefficient.

    Returns a dict of arrays ``lower``, ``upper`` and ``contains_dirac``
    (whether the Dirac mass at zero lies within the interval).
    """
    a = 1.0 - prob
    params = fit["parameters"]
    groups = np.asarray(params["groups"]).astype(int)
    grp_index = groups - groups.min()

    mu = np.asarray(fit["mu"], dtype=float)

    if not params["diag_covariance"]:
        s = np.concatenate([np.sqrt(np.diag(np.asarray(S, dtype=float)))
                            for S in fit["s"]])
    else:
        s = np.asarray(fit["s"], dtype=float)

    g = np.asarray(fit["g"], dtype=float)[grp_index]

    p = mu.size
    lower = np.zeros(p)
    upper = np.zeros(p)
    contains = np.zeros(p, dtype=bool)

    for i in range(p):
        gi, m, si = g[i], mu[i], s[i]

        if gi > 1 - a:
            ag = 1.0 - (1.0 - a) / gi
            lo, hi = norm.ppf([ag / 2, 1 - ag / 2], loc=m, scale=si)
            cd = False
            if lo <= 0 <= hi:
                lo, hi = norm.ppf(
                    [ag / 2 + (1 - gi) / 2, 1 - ag / 2 - (1 - gi) / 2],
                    loc=m, scale=si,
                )
                cd = True
            lower[i], upper[i], contains[i] = lo, hi, cd
        elif gi < a:
            lower[i], upper[i], contains[i] = 0.0, 0.0, True
        else:
            lo, hi = norm.ppf(
                [a / 2 + (1 - gi) / 2, 1 - a / 2 - (1 - gi) / 2],
                loc=m, scale=si,
            )
            lower[i], upper[i], contains[i] = lo, hi, True

    return {"lower": lower, "upper": upper, "contains_dirac": contains}
