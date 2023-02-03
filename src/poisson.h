#ifndef GSVB_FIT_POISSON_H
#define GSVB_FIT_POISSON_H

#include <vector>

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

// func for diag cov S
vec pois_update_mu(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const uvec &G, const vec &P);

vec pois_update_s(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const double lambda, const uvec &G, const vec &P);

double pois_update_g(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const double w, const uvec &G, 
	const vec &P);


// funcs for non diag S
vec pois_update_mu_S(const vec &yX_G, const mat &X_G, const vec &mu_G,
	const mat &U, const double lambda, const vec &P);

vec pois_update_U(const mat &X_G, const vec &mu_G, const mat &U,
	const double lambda, const vec &P);

double pois_update_g_S(const vec &yX_G, const mat &X_G, const vec &mu_G,
	const mat &U, const mat &S, const double lambda, const double w,
	const vec &P);

// ELBO
double elbo_poisson(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const vec &P,
	const double lambda, const double w, const uword mcn);

double elbo_poisson_S(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const std::vector<mat> &Us, const vec &g, 
	const vec &P, const double lambda, const double w, const uword mcn);

#endif
