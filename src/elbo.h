#ifndef GSVB_ELBO_H
#define GSVB_ELBO_H

#include "RcppArmadillo.h"
#include "gsvb_types.h"

double elbo(const vec &y, const mat &X, const uvec &groups, 
	const vec &mu, const vec &s, const vec &g, double lambda, 
	double a0, double b0, double sigma, uword mcn);

#endif
