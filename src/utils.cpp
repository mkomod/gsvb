#include "utils.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


vec sigmoid(const vec &x) {
    return 1/(1 + exp(-x));
}


vec mvnMGF(const mat &X, const mat &XX, const vec &mu, const vec &sig) 
{
    return exp(X * mu + 0.5 * XX * (sig % sig));
}


// [[Rcpp::export]]
vec mvnMGF(const mat &X, const vec &mu, const mat &S)
{
    const int n = X.n_rows;
    vec res = vec(n, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
	vec x = X.row(i).t();
	res(i) = exp(dot(x, mu) + 0.5 * dot(x, S * x));
    }

    return res;
}

// [[Rcpp::export]]
vec mvnMGF_chol(const mat &X, const vec &mu, const mat &U)
{
    const int n = X.n_rows;
    vec res = vec(n, arma::fill::zeros);

    for (int i = 0; i < n; ++i) {
	vec x = X.row(i).t();
	res(i) = exp(dot(x, mu) + 0.5 * pow(norm(U * x, 2), 2));
    }

    return res;
}


vec compute_P_G(const mat &X, const mat &XX, const vec &mu, const vec &s, const vec &g, 
	const uvec &G)
{
    return ((1 - g(G(0))) + g(G(0)) * mvnMGF(X.cols(G), XX.cols(G), mu(G), s(G)));
}


vec compute_P_G(const mat &X_G, const mat &XX_G, const vec &mu_G, const vec &s_G, const double g)
{
    return ((1 - g) + g * mvnMGF(X_G, XX_G, mu_G, s_G));
}


vec compute_P_G(const mat &X_G, const mat &mu_G, const mat &S, const double g) 
{
    return ((1.0 - g) + g * mvnMGF(X_G, mu_G, S));
}


vec compute_P_G_chol(const mat &X_G, const mat &mu_G, const mat &U, const double g)
{
    return ((1.0 - g) + g * mvnMGF_chol(X_G, mu_G, U));
}


vec compute_P(const mat &X, const mat &XX, const vec &mu, const vec &s, const vec &g, 
	const uvec &groups)
{
    vec P = vec(X.n_rows, arma::fill::ones);
    const uvec ugroups = unique(groups);

    for (uword group : ugroups) {
	uvec G = find(groups == group);
	P %= compute_P_G(X, XX, mu, s, g, G);
    }
    return P;
}


vec compute_P(const mat &X, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const uvec &groups)
{
    vec P = vec(X.n_rows, arma::fill::ones);
    const uvec ugroups = unique(groups);

    for (uword i = 0; i < ugroups.size(); ++i) {
	uvec G = find(groups == ugroups(i));
	P %= compute_P_G(X.cols(G), mu(G), Ss.at(i), g(G(0)));
    }
    return P;
}


vec compute_P_chol(const mat &X, const vec &mu, const std::vector<mat> &Us,
	const vec &g, const uvec &groups)
{
    vec P = vec(X.n_rows, arma::fill::ones);
    const uvec ugroups = unique(groups);

    for (uword i = 0; i < ugroups.size(); ++i) {
	uvec G = find(groups == ugroups(i));
	P %= compute_P_G_chol(X.cols(G), mu(G), Us.at(i), g(G(0)));
    }
    return P;
}
