#ifndef GSVB_UTILS_H
#define GSVB_UTILS_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"

double sigmoid(double x);
vec sigmoid(const vec &x);

vec mvnMGF(const mat &X, const mat &XX, const vec &mu, const vec &sig);

vec compute_P_G(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &G);

vec compute_P(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &groups);

#endif
