#ifndef GSVB_FIT_POISSON_H
#define GSVB_FIT_POISSON_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

// main funcs
vec pois_update_mu(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const uvec &G, const vec &P);

vec pois_update_s(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const double lambda, const uvec &G, const vec &P);

double pois_update_g(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const double w, const uvec &G, 
	const vec &P);

// ELBO
double elbo_poisson(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const vec &P,
	const double lambda, const double w, const uword mcn);

#endif
