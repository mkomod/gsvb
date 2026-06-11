"""Posterior predictive sampling. Port of R ``gsvb.predict``."""

import numpy as np

from .sample import gsvb_sample


def gsvb_predict(fit, newdata, samples=10000, quantiles=(0.025, 0.975),
                 return_samples=False, seed=None):
    """Sample from the posterior predictive distribution at ``newdata``.

    Returns a dict with ``mean`` and ``quantiles`` (rows = observations); if
    ``return_samples`` is True the raw draws are included under ``samples``.
    """
    rng = np.random.default_rng(seed)
    params = fit["parameters"]
    newdata = np.asarray(newdata, dtype=float)
    if newdata.ndim != 2:
        raise ValueError("newdata must be a 2-D matrix")

    if params["intercept"]:
        newdata = np.column_stack([np.ones(newdata.shape[0]), newdata])

    n = newdata.shape[0]
    draws = gsvb_sample(fit, samples=samples, seed=seed)
    Xb = newdata @ draws["beta"]                    # (n, samples)

    fam = params["family"]
    if fam == 1:
        from scipy.stats import t
        sigma = np.sqrt(fit["tau_b"] / fit["tau_a"])
        df = 2 + fit["tau_a"]
        y_star = Xb + sigma * t.rvs(df, size=Xb.shape, random_state=rng)
    elif fam in (2, 3, 4):
        y_star = 1.0 / (1.0 + np.exp(-Xb))
    else:  # poisson
        y_star = rng.poisson(np.exp(Xb)).astype(float)

    res = {
        "mean": y_star.mean(axis=1),
        "quantiles": np.quantile(y_star, q=np.asarray(quantiles), axis=1),
    }
    if return_samples:
        res["samples"] = y_star
    return res
