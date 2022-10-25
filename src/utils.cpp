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


vec compute_P_G(const mat &X, const mat &XX, const vec &mu, const vec &s, const vec &g, 
	const uvec &G)
{
    return ((1 - g(G(0))) + g(G(0)) * mvnMGF(X.cols(G), XX.cols(G), mu(G), s(G)));
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
