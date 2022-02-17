#include "elbo.h"


// [[Rcpp::export]]
double elbo(const vec &y, const mat &X, const uvec &groups, 
	const vec &mu, const vec &s, const vec &g, double lambda, 
	double a0, double b0, double sigma, uword mcn)
{
    const double n = X.n_rows;
    const uword p = X.n_cols;

    const double sigma_nsq = pow(sigma, -2.0);
    const mat xtx = X.t() * X;
    const double w = a0 / (a0 + b0);

    double res = - 0.5 * n * log(2 * M_PI * sigma * sigma) - 
	0.5 * sigma_nsq * dot(y, y) + 
	sigma_nsq * dot(y, X * (g % mu)) +
	0.5 * sum(g % log(2 * M_PI * pow(s, 2.0)));

    for (uword K : unique(groups).eval()) {
	uvec G = find(groups == K);
	uword k = G(0);
	double mk = G.size();
	
	double Ck = -mk*log(2.0) - 0.5*(mk-1.0)*log(M_PI) - lgamma(0.5*(mk+1));

	res += g(k) * Ck +
	    0.5 * g(k) * mk +
	    g(k) * mk * log(lambda) -
	    g(k) * log((1e-8 + g(k)) / (1e-8 + w)) -
	    (1 - g(k)) * log((1-g(k) + 1e-8) / (1 - w));
	
	// monte carlo integral
	double mci = 0.0;
	for (uword iter = 0; iter < mcn; ++iter) {
	    mci += norm(arma::randn(mk) % s(G) + mu(G), 2);
	}
	mci = mci / static_cast<double>(mcn);

	res -= lambda * g(k) * mci;
    }
    
    double a = 0.0;

    for (uword i = 0; i < p; ++i) {
	for (uword j = 0; j < p; ++j) {
	    if (i == j) {
		a += xtx(j, j) * g(j) * (s(j) * s(j) + mu(j) * mu(j));
	    }
	    if (i != j && groups(i) == groups(j)) {
		a += xtx(j, i) * g(j) * mu(j) * mu(i); 
	    }
	    if (i != j && groups(i) != groups(j)) {
		a += xtx(j, i) * g(j) * g(i) * mu(j) * mu(i);
	    }
	}
    }

    res -= 0.5 * sigma_nsq * a;

    return(res);
}






