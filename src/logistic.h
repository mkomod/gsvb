#ifndef GSVB_FIT_LOGISTIC_H
#define GSVB_FIT_LOGISTIC_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

// expected log-likelihood introduced for the new bound
double ell(const mat &Xm, const mat &Xs,  const vec &g, double thresh, int l);
double tll(const vec &mu, const vec &sig, const int l);

vec update_m(const vec &y, const mat &X, const vec &m, const vec &s, const vec &ug,
	double lambda, uword group, const uvec G, mat &Xm, const mat &Xs, 
	const double thresh, const int l);

vec update_s(const vec &y, const mat &X, const vec &m, const vec &s, vec ug, 
	const double lambda, const uword group, const uvec G, 
	const mat &Xm, mat &Xs, const double thresh, const int l);

double update_g(const vec &y, const mat &X, const vec &m, const vec &s, 
	vec ug, const double lambda, const uword group,
	const uvec &G, const mat &Xm, const mat &Xs,
	const double thresh, const int l, const double w);


#endif
