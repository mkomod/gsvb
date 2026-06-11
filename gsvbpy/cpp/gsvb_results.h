#ifndef GSVB_RESULTS_H
#define GSVB_RESULTS_H

#include <vector>
#include "gsvb_types.h"

// Plain result structs returned by the fit functions, replacing the
// Rcpp::List returns of the original code. The pybind11 binding layer
// (bindings.cpp) converts these into Python dicts with the same field names
// the R package used.

struct LinearFit {
    vec mu;
    vec sigma;
    std::vector<mat> S;
    vec gamma;
    double tau_a;
    double tau_b;
    bool converged;
    uword iterations;
    std::vector<double> elbo;
};

// Shared by the logistic and poisson fits (no tau term).
struct GlmFit {
    vec mu;
    vec sigma;
    vec gamma;
    bool converged;
    uword iterations;
    std::vector<mat> S;
    std::vector<double> elbo;
};

#endif
