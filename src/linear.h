#ifndef GSVB_FIT_LINEAR_H
#define GSVB_FIT_LINEAR_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

vec update_mu(const uvec &G, const uvec &Gc, const mat &xtx, 
	const vec &yx, const vec &mu, const vec &s, const vec &g, 
	const double sigma, const double lambda);

vec update_s(const uvec &G, const mat &xtx, const vec &mu, 
	const vec &s, const double sigma, const double lambda);

vec update_S(const uvec &G, const mat &xtx, const vec &mu, 
	mat &S, vec s, const double e_tau, const double lambda);

double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const vec &s, const vec &g, double sigma,
	double lambda, double w);

double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const mat &S, const vec &g, double e_tau,
	double lambda, double w);

void update_a_b(double &tau_a, double &tau_b, const double tau_a0,
	const double tau_b0, const double S, const double n);

double compute_R(double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const uword p, const bool approx,
	const double approx_thresh=1e-3);

double compute_R(const double yty, const vec &yx, const mat &xtx,
	const uvec &groups, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const uword p, const bool approx, 
	const double approx_thresh=1e-3);

double elbo_linear_c(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const vec &s, const vec &g,
	const double tau_a, const double tau_b, const double lambda, 
	const double a0, const double b0, const double tau_a0, 
	const double tau_b0, const uword mcn, const bool approx, 
	const double approx_thresh=1e-3);

double elbo_linear_u(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const double tau_a, const double tau_b, const double lambda,
	const double a0, const double b0, const double tau_a0, const double tau_b0, 
	const uword mcn, const bool approx, const double approx_thresh=1e-3);

#endif
