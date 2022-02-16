#include "fit.h"


// [[Rcpp::export]]
Rcpp::List fit(vec y, mat X, uvec groups, const double lambda, const double a0,
	const double b0, const double sigma, vec mu, vec s, vec g, 
	unsigned int niter, double tol, bool verbose)
{
    const uword n = X.n_rows;
    const uword p = X.n_cols;
    const double w = a0 / (a0 + b0);

    const mat xtx = X.t() * X;
    const vec yx = (y.t() * X).t();
    
    const uvec ugroups = arma::unique(groups);
    vec mu_old, s_old, g_old;

    // init s
    s = arma::pow(xtx.diag() / pow(sigma, 2.0) + 2.0 * lambda, -0.5);
    uword num_iter = niter;
    bool converged = false;

    for (unsigned int iter = 1; iter <= niter; ++iter) {

	mu_old = mu; s_old = s; g_old = g;
	
	for (uword group : ugroups) {
	    uvec G  = arma::find(groups == group);
	    uvec Gc = arma::find(groups != group);

	    mu(G) = update_mu(G, Gc, xtx, yx, mu, s, g, sigma, lambda);
	    s(G)  = update_s(G, xtx, mu, s, sigma, lambda);
	    double tg = update_g(G, Gc, xtx, yx, mu, s, g, sigma, lambda, w);
	    for (uword j : G) g(j) = tg;
    
	}
	
	// check from break
	Rcpp::checkUserInterrupt();
	if (verbose) Rcpp::Rcout << iter;

	// check convergence
	if (sum(abs(mu_old - mu)) < tol &&
	    sum(abs(s_old - s))   < tol &&
	    sum(abs(g_old - g))   < tol) 
	{
	    if (verbose)
		Rcpp::Rcout << "\nConverged in " << iter << " iterations\n";
	    num_iter = iter;
	    converged = true;
	    break;
	}
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = converged,
	Rcpp::Named("iterations") = num_iter
    );
}

// ----------------- mu -------------------
class update_mu_fn
{
    public:
	update_mu_fn(const uvec &G, const uvec &Gc, const mat &xtx, 
		const vec &yx, const vec &mu, const vec &s, const vec &g, 
		const double sigma, const double lambda) :
	    G(G), Gc(Gc), xtx(xtx), yx(yx), mu(mu), s(s), g(g), 
	    sigma(sigma), lambda(lambda)
	    { }

	double EvaluateWithGradient(const arma::mat &m, arma::mat &grad) {
	    const double sigma_s = pow(sigma, -2.0);
	    const double res = 0.5 * sigma_s * dot(m.t() * xtx(G, G), m) + 
		sigma_s * dot(m.t() * xtx(G, Gc), (g(Gc) % mu(Gc))) -
		sigma_s * dot(yx(G), m) +
		lambda * pow(dot(s(G), s(G)) + dot(m, m), 0.5);
	    grad = sigma_s * xtx(G, G) * m + 
		sigma_s * xtx(G, Gc) * (g(Gc) % mu(Gc)) -
		sigma_s * yx(G) +
		lambda * m * pow(dot(s(G), s(G)) + dot(m, m), -0.5);
	    return res;
	}

    private:
	const uvec &G;
	const uvec &Gc;
	const mat &xtx;
	const vec &yx;
	const vec &mu;
	const vec &s;
	const vec &g;
	const double sigma;
	const double lambda;
};


// [[Rcpp::export]]
vec update_mu(const uvec &G, const uvec &Gc, const mat &xtx, 
	const vec &yx, const vec &mu, const vec &s, const vec &g, 
	const double sigma, const double lambda)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 5;
    update_mu_fn fn(G, Gc, xtx, yx, mu, s, g, sigma, lambda);

    vec m = mu(G);
    opt.Optimize(fn, m);

    return m;
}


// [[Rcpp::export]]
double update_mu_fn_2(const vec &m, const mat &xtx, const vec &yx, const vec &mu, 
	const vec &s, const vec &g, const double sigma, const double lambda, 
	const uvec &G, const uvec &Gc, const uword mcn)
{
    const double sigma_s = pow(sigma, -2.0);
    double mci = 0.0;
    for (uword iter = 0; iter < mcn; ++iter) {
	mci += norm(arma::randn(size(m)) % s(G) + m, 2);
    }
    mci = mci / static_cast<double>(mcn);

    const double res = 0.5 * sigma_s * dot(m.t() * xtx(G, G), m) + 
	sigma_s * dot(m.t() * xtx(G, Gc), (g(Gc) % mu(Gc))) -
	sigma_s * dot(yx(G), m) +
	lambda * mci;

    return res;
}


// ----------------- sigma -------------------
class update_s_fn
{
    public:
	update_s_fn(const uvec &G, const mat &xtx, const vec &mu, 
		const double sigma, const double lambda) :
	    G(G), xtx(xtx), mu(mu), sigma(sigma), lambda(lambda) { }

	double EvaluateWithGradient(const arma::mat &s, arma::mat &grad) {
	    const double sigma_s = pow(sigma, -2.0);
	    const double res = 0.5 * sigma_s * dot(diagvec(xtx(G, G)), s % s) -
		accu(log(s)) + lambda * pow(dot(s, s) + dot(mu(G), mu(G)), 0.5);
	    grad = 0.5 * sigma_s * diagvec(xtx(G, G)) % s -
		1/s + lambda * s * pow(dot(s, s) + dot(mu(G), mu(G)), -0.5);
	    return res;
	}

    private:
	const uvec &G;
	const mat &xtx;
	const vec &mu;
	const double sigma;
	const double lambda;
};


// [[Rcpp::export]]
vec update_s(const uvec &G, const mat &xtx, const vec &mu, 
	const vec &s, const double sigma, const double lambda)
{
    ens::L_BFGS opt;
    update_s_fn fn(G, xtx, mu, sigma, lambda);
    opt.MaxIterations() = 5;
    
    vec sig = s(G);
    opt.Optimize(fn, sig);

    return abs(sig);
}


// ----------------- gamma -------------------
// [[Rcpp::export]]
double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const vec &s, const vec &g, double sigma,
	double lambda, double w)
{
    const double mk = G.size();
    const double sigma_s = pow(sigma, -2.0);
    double res = log(w / (1.0 - w)) + mk/2.0 + sigma_s * arma::dot(yx(G), mu(G)) +
	0.5 * mk * log(2.0 * M_PI) +
	sum(log(s(G))) -
	mk * log(2.0) - 0.5 * (mk - 1.0) * log(M_PI) - lgamma(0.5 * (mk + 1)) +
	mk * log(lambda) - 
	lambda * sqrt(sum(pow(s(G), 2.0)) + sum(pow(mu(G), 2.0))) -
	0.5 * sigma_s * dot(diagvec(xtx(G, G)), pow(s(G), 2.0)) -
	0.5 * sigma_s * dot(mu(G).t() * xtx(G, G), mu(G)) -
	sigma_s * dot(mu(G).t() * xtx(G, Gc), g(Gc) % mu(Gc));

    return sigmoid(res);
}

