#ifndef GSVB_FIT_H
#define GSVB_FIT_H

#include "RcppEigen.h"

#include "gsvb_types.h"
#include "utils.h"

double update_mu(unsigned int i, int gi, const mat &xtx, double yx_i, 
	const vec &mu, const vec &s, const vec &g, double sigma, double lambda,
	const std::vector<std::array<int, 2>> &gindices);

double update_g(unsigned int gi, const mat &xtx, const vec &yx, const vec &mu,
	const vec &s, const vec &g, double sigma, double lambda, double w,
	const std::vector<std::array<int, 2>> &gindices);

#endif
