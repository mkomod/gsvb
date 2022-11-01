#include "poisson.h"

#define GSVB_POS_MAXITS 8

// [[Rcpp::export]]
Rcpp::List fit_poisson(vec y, mat X, uvec groups, const double lambda, 
    const double a0, const double b0, vec mu, vec s, vec g, 
    const bool diag_cov, bool track_elbo, const uword track_elbo_every, 
    const uword track_elbo_mcn, unsigned int niter, double tol, bool verbose)
{
    const uvec ugroups = arma::unique(groups);
    const double w = a0 / (a0 + b0);
    const mat XX = X % X;
    const vec yX = X.t() * y;
    
    // init
    vec mu_old, s_old, g_old;
    vec P = compute_P(X, XX, mu, s, g, groups);

    uword num_iter = niter;
    std::vector<double> elbo_values;
    bool converged = false;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
	mu_old = mu; s_old = s; g_old = g;

	for (uword group : ugroups)
	{
	    uvec G  = arma::find(groups == group);
	    P /= compute_P_G(X, XX, mu, s, g, G);

	    mu(G) = pois_update_mu(yX, X, XX, mu, s, lambda, G, P);
	    s(G)  = pois_update_s(     X, XX, mu, s, lambda, G, P);

	    double tg = pois_update_g(yX, X, XX, mu, s, lambda, w, G, P);
	    for (uword j : G) g(j) = tg;

	    P %= compute_P_G(X, XX, mu, s, g, G);
	}

	if (track_elbo && (iter % track_elbo_every == 0)) {
	    double e = elbo_poisson(y, X, groups, mu, s, g, P, lambda, w, 
		    track_elbo_mcn);
	    elbo_values.push_back(e);
	}
	
	// check for break, print iter
	Rcpp::checkUserInterrupt();
	if (verbose) Rcpp::Rcout << iter;
	
	// check convergence
	if (sum(abs(mu_old - mu)) < tol &&
	    sum(abs(s_old  - s))  < tol &&
	    sum(abs(g_old  - g))  < tol) 
	{
	    if (verbose)
		Rcpp::Rcout << "\nConverged in " << iter << " iterations\n";

	    num_iter = iter;
	    converged = true;
	    break;
	}
    }
    
    // compute elbo for final eval
    if (track_elbo) {
	double e = elbo_poisson(y, X, groups, mu, s, g, P, lambda, w, 
		track_elbo_mcn);
	elbo_values.push_back(e);
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = converged,
	Rcpp::Named("iterations") = num_iter,
	Rcpp::Named("elbo") = elbo_values
    );
}


// --------- update mu ----------
class pois_update_mu_fn
{
    public:
	pois_update_mu_fn(const vec &yX, const mat &X, const mat &XX,
		const vec &s, const double lambda, const uvec &G,
		const vec &P) :
	    yX(yX), X(X), XX(XX), s(s), lambda(lambda), G(G), P(P)
	{};

	double EvaluateWithGradient(const mat &mG, mat &grad)
	{
	    const vec PP = P % mvnMGF(X.cols(G), XX.cols(G), mG, s(G));

	    double res = - dot(yX(G), mG) +
		accu(PP) +
		lambda * sqrt(accu(s(G) % s(G) + mG % mG)); 

	    vec dPPmG = vec(mG.size(), arma::fill::zeros);

	    for (uword j = 0; j < mG.size(); ++j) {
		dPPmG(j) = accu(X.col(G(j)) % PP);
	    }
	    
	    grad = dPPmG -
		yX(G) +
		lambda * mG * pow(dot(s(G), s(G)) + dot(mG, mG), -0.5);
	    
	    return res;
	};

    private:
	const vec &yX;
	const mat &X;
	const mat &XX;
	const vec &s;
	const double lambda;
	const uvec &G;
	const vec &P;
};


vec pois_update_mu(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const uvec &G, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = GSVB_POS_MAXITS;
    pois_update_mu_fn fn(yX, X, XX, s, lambda, G, P);

    arma::vec mG = mu(G);
    opt.Optimize(fn, mG);

    return mG;
}


// --------- update s ----------
class pois_update_s_fn
{
    public:
	pois_update_s_fn(const mat &X, const mat &XX, const vec &mu,
		const double lambda, const uvec &G, const vec &P) :
	    X(X), XX(XX), mu(mu), lambda(lambda), G(G), P(P)
	{};

	double EvaluateWithGradient(const mat &u, mat &grad)
	{
	    const vec sG = exp(u);

	    const vec PP = P % mvnMGF(X.cols(G), XX.cols(G), mu(G), sG);

	    double res = accu(PP) -
		accu(log(sG)) +
		lambda * sqrt(accu(sG % sG + mu(G) % mu(G)));

	    vec dPPsG = vec(sG.size(), arma::fill::zeros);

	    for (uword j = 0; j < sG.size(); ++j) {
		dPPsG(j) = sG(j) * accu((XX.col(G(j)) % PP));
	    }
	    
	    // df/duG = df/dsG * dsG/du
	    grad = (dPPsG -
		1.0 / sG +
		lambda * sG * pow(dot(sG, sG) + dot(mu(G), mu(G)), -0.5)) % sG;
	    
	    return res;
	};

    private:
	const mat &X;
	const mat &XX;
	const vec &mu;
	const double lambda;
	const uvec &G;
	const vec &P;
};


vec pois_update_s(const mat &X, const mat &XX, const vec &mu, const vec &s,
	const double lambda, const uvec &G, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = GSVB_POS_MAXITS;
    pois_update_s_fn fn(X, XX, mu, lambda, G, P);

    arma::vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


// --------- update g ----------
double pois_update_g(const vec &yX, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const double w, const uvec &G, 
	const vec &P)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    const vec P1 = mvnMGF(X.cols(G), XX.cols(G), mu(G), s(G));

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * accu(log(2.0 * M_PI * s(G) % s(G))) -
	lambda * sqrt(dot(s(G), s(G)) + dot(mu(G), mu(G))) +
	dot(yX(G), mu(G)) -
	sum(P % (P1 - 1));

    return 1.0/(1.0 + exp(-res));
}


// ---------------------------------------
// ELBO
// ---------------------------------------
// [[Rcpp::export]]
double elbo_poisson(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const double lambda, 
	const double w, const uword mcn)
{
    const mat &XX = X % X;
    const vec P = compute_P(X, XX, mu, s, g, groups);
    double res = elbo_poisson(y, X, groups, mu, s, g, P, lambda, w, mcn);

    return(res);
}


double elbo_poisson(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const vec &P,
	const double lambda, const double w, const uword mcn)
{
    double res = 0.0;
    uvec ugroups = arma::unique(groups);
    
    res += dot(y, (X * (mu % g))) - accu(P) - accu(lgamma(y + 1));

    // noramlizing consts
    for (uword group : ugroups) 
    {
	uvec G = find(groups == group);
	uword k = G(0);

	double mk = G.size();
	double Ck = -mk*log(2.0) - 0.5*(mk-1.0)*log(M_PI) - lgamma(0.5*(mk+1));

	res += g(k) * Ck +
	    0.5 * g(k) * mk +
	    g(k) * mk * log(lambda) -
	    g(k) * log((1e-8 + g(k)) / (1e-8 + w)) -	// add 1e-8 to prevent -Inf
	    (1 - g(k)) * log((1-g(k) + 1e-8) / (1 - w));
    }

    // monte carlo integral for intractable terms
    double mci = 0.0;
    vec beta = vec(mu.n_rows, arma::fill::zeros);

    for (uword iter = 0; iter < mcn; ++iter) {
	for (uword group : ugroups)
	{
	    uvec G = find(groups == group);
	    uword k = G(0);
	    double mk = G.size();

	    // Compute the Monte-Carlo integral of E_Q [ lambda * || b_{G_k} || ]
	    vec beta_G = arma::randn(mk) % s(G) + mu(G);
	    mci -= lambda * g(k) * norm(beta_G, 2);
	}
    }
    mci = mci / static_cast<double>(mcn);
    res += mci;

    return(res);
}
