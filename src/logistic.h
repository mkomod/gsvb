#ifndef GSVB_FIT_LOGISTIC_H
#define GSVB_FIT_LOGISTIC_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"
#include "utils.h"

// expected log-likelihood introduced for the new bound
double ell(const mat &Xm, const mat &Xs,  const vec &g, double thresh, int l);
double tll(const vec &mu, const vec &sig, const int l);

vec nb_update_m(const vec &y, const mat &X, const vec &m, const vec &s, const vec &ug,
	double lambda, uword group, const uvec G, mat &Xm, const mat &Xs, 
	const double thresh, const int l);

vec nb_update_s(const vec &y, const mat &X, const vec &m, const vec &s, vec ug, 
	const double lambda, const uword group, const uvec G, 
	const mat &Xm, mat &Xs, const double thresh, const int l);

double nb_update_g(const vec &y, const mat &X, const vec &m, const vec &s, 
	vec ug, const double lambda, const uword group,
	const uvec &G, const mat &Xm, const mat &Xs,
	const double thresh, const int l, const double w);


// jensens functions
vec jen_update_mu(const vec &yX_G, const mat &X_G, const mat &XX_G, 
	const vec &mu_G, const vec &s_G, const double lambda, const vec &P);

vec jen_update_s(const mat &X_G, const mat &XX_G, const vec &mu_G,
	const vec &s_G, const double lambda, const vec &P);

double jen_update_g(const vec &yX_G, const mat &X_G, const mat &XX_G, const vec &mu_G,
	const vec &s_G, const double lambda, const double w, const double mk, 
	const vec &P);

// jaakkola functions
vec jaak_update_mu(const vec &y, const mat &X, const mat &XAX,
	const vec &mu, const vec &s, const vec &g, const double lambda,
	const uvec &G, const uvec &Gc);

vec jaak_update_s(const mat &XAX, const vec &mu, 
	const vec &s, const double lambda, const uvec &G);

vec jaak_update_S(const mat &XAX, const vec &mu, mat &S, const vec &s, 
	const double lambda, const uvec &G);

double jaak_update_g(const vec &y, const mat &X, const mat &XAX,
	const vec &mu, const vec &s, const vec &g, const double lambda,
	const double w, const uvec &G, const uvec &Gc);


// uses S not sigma^2, this is for full covaraince
double jaak_update_g(const vec &y, const mat &X, const mat &XAX, const vec &mu,
	const mat &S, const vec &g, const double lambda, const double w,
	const uvec &G, const uvec &Gc);

vec jaak_update_l(const mat &X, const vec &mu, const vec &s, const vec &g);

vec jaak_update_l(const mat &X, const vec &mu, const std::vector<mat> &Ss,
	const vec &g, const uvec &groups, const uvec &ugroups);


// jaakola helper
vec a(const vec &x);


// ELBO
double elbo_logistic(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const std::vector<mat> &Ss,
	const double lambda, const double w, const uword mcn, const bool diag);

#endif
