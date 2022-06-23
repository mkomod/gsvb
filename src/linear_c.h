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

void update_a_b(double &tau_a, double &tau_b, const double tau_a0,
	const double tau_b0, const double S, const double n);

double compute_S(double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const uword p, const bool approx,
	const double approx_thresh=1e-3);

double elbo(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const vec &s, const vec &g,
	const double tau_a, const double tau_b, const double lambda, 
	const double a0, const double b0, const double tau_a0, 
	const double tau_b0, const uword mcn, const bool approx, const double
	approx_thresh=1e-3);

#endif
