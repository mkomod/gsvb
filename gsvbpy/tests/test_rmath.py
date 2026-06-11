"""Validate the self-contained special functions against SciPy references."""

import numpy as np
import pytest

from scipy import special, stats

import gsvbpy._gsvb_core as core


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 1.5, 3.0, 5.9, 6.0, 6.1, 20.0, 1e3])
def test_digamma(x):
    assert core._digamma(x) == pytest.approx(special.digamma(x), rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 1.5, 3.0, 5.9, 6.0, 6.1, 20.0, 1e3])
def test_trigamma(x):
    assert core._trigamma(x) == pytest.approx(special.polygamma(1, x),
                                              rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("x", [-40.0, -8.0, -1.0, 0.0, 1.0, 8.0, 40.0])
def test_pnorm_plain(x):
    # lower tail, no log
    assert core._pnorm(x, 0.0, 1.0, 1, 0) == pytest.approx(stats.norm.cdf(x), abs=1e-12)
    # upper tail, no log
    assert core._pnorm(x, 0.0, 1.0, 0, 0) == pytest.approx(stats.norm.sf(x), abs=1e-12)


@pytest.mark.parametrize("x", [-40.0, -8.0, -1.0, 0.0, 1.0, 8.0, 40.0, 100.0])
def test_pnorm_log(x):
    # stable log-tail path -- the numerically critical case
    assert core._pnorm(x, 0.0, 1.0, 1, 1) == pytest.approx(stats.norm.logcdf(x),
                                                          rel=1e-9, abs=1e-9)
    assert core._pnorm(x, 0.0, 1.0, 0, 1) == pytest.approx(stats.norm.logsf(x),
                                                          rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 1.0, 5.0])
def test_dnorm(x):
    assert core._dnorm(x, 0.0, 1.0, 0) == pytest.approx(stats.norm.pdf(x), abs=1e-12)
    assert core._dnorm(x, 0.5, 2.0, 1) == pytest.approx(stats.norm.logpdf(x, 0.5, 2.0),
                                                       rel=1e-12, abs=1e-12)
