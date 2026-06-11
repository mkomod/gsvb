"""gsvbpy: group-sparse variational Bayes regression (Python port of GSVB).

Scalable group-sparse Bayesian linear, logistic (Jensen, Jaakkola and refined
bounds) and Poisson regression with uncertainty quantification. The variational
inference runs in a compiled C++ backend (Armadillo + Ensmallen).
Reference: https://arxiv.org/abs/2309.10378
"""

from . import _gsvb_core
from .fit import gsvb_fit
from .predict import gsvb_predict
from .sample import gsvb_sample
from .credible_intervals import gsvb_credible_intervals
from .elbo import gsvb_elbo


def set_seed(seed):
    """Seed the C++ backend's RNG (used by the Monte-Carlo ELBO terms)."""
    _gsvb_core.set_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)


__all__ = [
    "gsvb_fit",
    "gsvb_predict",
    "gsvb_sample",
    "gsvb_credible_intervals",
    "gsvb_elbo",
    "set_seed",
]

__version__ = "0.1.0"
