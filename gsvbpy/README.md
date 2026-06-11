# gsvbpy

Group-sparse variational Bayes regression — a Python port of the
[GSVB](https://github.com/mkomod/gsvb) R package.

GSVB is software for scalable group-sparse regression. Unlike other
state-of-the-art group-selection methods, it provides scalable uncertainty
quantification (~100x faster than MCMC). Currently the linear, logistic and
Poisson models are available.

The compute-heavy backend is written in C++ (using
[Armadillo](https://arma.sourceforge.net/) for linear algebra and
[Ensmallen](https://ensmallen.org/) for L-BFGS optimization) and is bound to
Python with [pybind11](https://pybind11.readthedocs.io/). Armadillo and
Ensmallen are vendored under `third_party/`; the only external runtime
dependency for the math kernels is a BLAS/LAPACK library (OpenBLAS).

## Install

Requires a C++ compiler, CMake (>= 3.18) and a BLAS/LAPACK library. On
Debian/Ubuntu:

```bash
sudo apt-get install -y libopenblas-dev
pip install .
```

## Example

```python
import numpy as np
import gsvbpy

rng = np.random.default_rng(1)

n, p, gsize = 100, 1000, 5
groups = np.repeat(np.arange(1, p // gsize + 1), gsize)

X = rng.standard_normal((n, p))
b = np.concatenate([np.zeros(gsize), np.full(gsize, -4.0),
                    np.full(gsize, 8.0), np.zeros(p - 3 * gsize)])
y = X @ b + rng.standard_normal(n)

gsvbpy.set_seed(1)
f = gsvbpy.gsvb_fit(y, X, groups, family="gaussian")

print("converged:", f["converged"], "in", f["iter"], "iterations")
# f["beta_hat"] holds the variational posterior mean
```

## API

- `gsvb_fit(y, X, groups, family=...)` — fit the model. Families:
  `"gaussian"`, `"binomial-jensens"`, `"binomial-jaakkola"`,
  `"binomial-refined"`, `"poisson"`. Pass `diag_covariance=False` for a
  per-group full covariance (supported for `"gaussian"`,
  `"binomial-jaakkola"` and `"poisson"`; the Jensen and refined bounds are
  diagonal only).
- `gsvb_predict(fit, newdata, ...)` — posterior predictive samples.
- `gsvb_sample(fit, samples=...)` — draw from the variational posterior.
- `gsvb_credible_intervals(fit, prob=...)` — marginal credible intervals.
- `gsvb_elbo(fit, y, X, ...)` — evidence lower bound.
- `set_seed(seed)` — seed the backend RNG.

## Notes on the port

- Initialization of `mu` uses scikit-learn (`Ridge`/`Lasso`/`LogisticRegression`/
  `PoissonRegressor`) in place of the R package's `gglasso`/`glmnet` fits.
  Initialization only affects convergence speed, not the model.
- The R package's RNG (`RNGScope`) is replaced by Armadillo's RNG, so
  Monte-Carlo ELBO values differ from R within Monte-Carlo error; the
  deterministic fit outputs (`mu`, `s`, `g`) match.

## Reference

Komodromos et al., *Group sparse Bayesian regression via variational
inference*. <https://arxiv.org/abs/2309.10378>
