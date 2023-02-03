#ifndef GSVB_UTILS_H
#define GSVB_UTILS_H

#include <vector>

#include "RcppEnsmallen.h"
#include "gsvb_types.h"

double sigmoid(double x);

vec sigmoid(const vec &x);

vec mvnMGF(const mat &X, const mat &XX, const vec &mu, const vec &sig);

vec mvnMGF(const mat &X, const vec &mu, const mat &S);

vec mvnMGF_chol(const mat &X, const vec &mu, const mat &U);

vec compute_P_G(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &G);

vec compute_P_G(const mat &X_G, const mat &XX_G, const vec &mu_G, 
	const vec &s_G, const double g);

vec compute_P_G(const mat &X_G, const mat &mu_G, const mat &S, 
	const double g);

vec compute_P_G_chol(const mat &X_G, const mat &mu_G, const mat &U,
	const double g);

vec compute_P(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const vec &g, const uvec &groups);

vec compute_P(const mat &X, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const uvec &groups);

vec compute_P_chol(const mat &X, const vec &mu, const std::vector<mat> &Us,
	const vec &g, const uvec &groups);

#endif
