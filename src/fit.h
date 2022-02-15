#ifndef GSVB_FIT_H
#define GSVB_FIT_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

vec update_mu(const uvec &G, const uvec &Gc, const mat &xtx, 
	const vec &yx, const vec &mu, const vec &s, const vec &g, 
	const double sigma, const double lambda);

vec update_s(const uvec &G, const mat &xtx, const vec &mu, 
	const vec &s, const double sigma, const double lambda);

double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const vec &s, const vec &g, double sigma,
	double lambda, double w);

#endif
