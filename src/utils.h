#ifndef GSVB_UTILS_H
#define GSVB_UTILS_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"

double sigmoid(double x);
vec sigmoid(const vec &x);

vec mvnMGF(const mat &X, const mat &XX, const vec &mu, const vec &sig);

vec compute_P_G(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &G);

vec compute_P_G(const mat &X_G, const mat &XX_G, const vec &mu_G, 
	const vec &s_G, const double g);

vec compute_P(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &groups);


#endif
