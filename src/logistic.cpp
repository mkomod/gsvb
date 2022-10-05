#include "logistic.h"
#include <bitset>



// [[Rcpp::export]]
Rcpp::List fit_logistic(vec y, mat X, uvec groups, const double lambda, const double a0,
    const double b0, vec mu, vec s, vec g, const double thresh, const int l,
    unsigned int niter, double tol, bool verbose)
{
    const uword n = X.n_rows;
    const uword p = X.n_cols;
    const double w = a0 / (a0 + b0);
    
    // init
    const uvec ugroups = arma::unique(groups);
    vec mu_old, s_old, g_old;
    mat Xm = mat(n, ugroups.size());
    mat Xs = mat(n, ugroups.size());

    for (uword group : ugroups) 
    {
	uvec G = arma::find(groups == group);
	Xm.col(group) = X.cols(G) * mu(G);
	Xs.col(group) = (X.cols(G) % X.cols(G)) * (s(G) % s(G));
    }
    
    uword num_iter = niter;
    bool converged = false;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
	mu_old = mu; s_old = s; g_old = g;

	// update mu, sigma, gamma
	for (uword group : ugroups)
	{
	    uvec G  = arma::find(groups == group);
	    uvec Gc = arma::find(groups != group);
	    
	    // TODO: there's a bug in update_m
	    mu(G) = update_m(y, X, mu, s, g, lambda, group, G, Xm, Xs, thresh, l);
	    s(G) = update_s(y, X, mu, s, g, lambda, group, G, Xm, Xs, thresh, l);

	    double tg = update_g(y, X, mu, s, g, lambda, group, G, Xm, Xs, 
		    thresh, l, w);
	    for (uword j : G) g(j) = tg;
	}
	
	// check for break, print iter
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
    
    // compute elbo for final eval
    // if (track_elbo) {
	// double e = diag_cov ?
	    // elbo_linear_c(yty, yx, xtx, groups, n, p, mu, s, g, tau_a, tau_b,
		// lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false) :
	    // elbo_linear_u(yty, yx, xtx, groups, n, p, mu, Ss, g, tau_a, tau_b,
		// lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false);
	// elbo_values.push_back(e);
    // }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = converged,
	Rcpp::Named("iterations") = num_iter
	// Rcpp::Named("elbo") = elbo_values
    );
}


// ----------- expected log likelihood ----------
double tll(const vec &mu, const vec &sig, const int l)
{
    // computes an upper bound for
    // E_X [log(1 + exp(X)) ] where X ~ N(mu, sig)
    const int n = mu.n_rows;
    double res = 0.0;

    for (int i = 0; i < n; ++i) 
    {
	double a = sig(i) / sqrt(2.0 * M_PI) * 
	    exp(- 0.5 * mu(i)*mu(i) / (sig(i)*sig(i))) +
	    mu(i) * R::pnorm(mu(i) / sig(i), 0, 1, 1, 0);

	double b = 0.0;
	for(int j = 1; j <= (2 * l - 1); ++j) {
	    b += pow((-1.0), (j-1)) / j * (
		exp(
		    mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(-mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		) +
		exp(
		    -mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		)
	    );
	}

	res += a + b;
    }
    
    return res;
}


double ell(const mat &Xm, const mat &Xs,  const vec &g, double thresh, int l)
{
    // computes an upper bound for the expected log-likelihood
    // E_Q [ log(1 + exp(x' b) ] where b ~ Q = SpSL = gN + (1-g) delta_0
    // 
    // The function thresholds values of g to compute the expecation
    //
    const uvec mid = find(g >= thresh && g <= (1.0 - thresh));
    const uvec big = find(g > (1.0 - thresh));
    const int msize = mid.size();
    
    // compute mu and sig
    const vec mu = sum(Xm.cols(big), 1);
    const vec sig = sum(Xs.cols(big), 1);

    double res = 0.0;
    if (msize == 0) 
    {
	res = tll(mu, sqrt(sig), l);
    } 
    else 
    {
	double tot = 0.0;
	for (int i = 0; i < pow(2, msize); ++i) 
	{
	    auto b = std::bitset<12>(i);
	    double prod_g = 1.0;

	    vec mu_new = mu;
	    vec sig_new = sig;

	    for (int j = 0; j < msize; ++j) 
	    {
		if ((i >> j) & 1) 
		{
		    mu_new += Xm.col(mid(j));
		    sig_new += Xs.col(mid(j));
		    prod_g *= g(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - g(mid(j)));
		}
	    }
	    
	    tot += prod_g * tll(mu_new, sqrt(sig_new), l);
	}

	res = tot;
    }

    return  res;
}


// ------- gradients wrt. m ---------
vec dt_dm(const mat &X, const vec &mu, const vec &sig, const uvec &G, 
	const int l) 
{
    // gradient wrt. mu
    const int n = mu.n_rows;
    const int mk = G.n_rows;
    vec res = arma::vec(mk, arma::fill::zeros);
    
    // dt/dm = dt/dmu dmu/dm
    //       = dt/dmu * X_{i, G}
    for (int i = 0; i < n; ++i) 
    {
	double dt_dmu = 0.0;

	dt_dmu += sig(i) / sqrt(2.0 * M_PI) * 
	    - mu(i) / (sig(i)*sig(i)) * exp(- 0.5 * mu(i)*mu(i)/(sig(i)*sig(i))) +
	    R::pnorm(mu(i) / sig(i), 0, 1, 1, 0) +
	    mu(i)/sig(i) * R::dnorm4(mu(i) / sig(i), 0, 1, 0);
	
	for(int j = 1; j <= (2 * l - 1); ++j) 
	{
	    dt_dmu += pow((-1.0), (j-1)) / j * (
		j * exp(
		    mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(-mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		) + 
		- j * exp(
		    -mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		) +
		- 1.0/sig(i) * exp(
		    mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::dnorm(-mu(i)/sig(i) - j*sig(i), 0, 1, 1)
		) +
		1.0/sig(i) * exp(
		    -mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::dnorm(mu(i)/sig(i) - j*sig(i), 0, 1, 1)
		)
	    );
	}
	
	// apply the chain rule
	for (int k = 0; k < mk; ++k) {
	    res(k) += dt_dmu * X(i, G(k));
	}
    }

    return res;
}


vec dell_dm(const mat &X, const mat &Xm, const mat &Xs, const vec &g,
	const uvec &G, const double thresh, const int l)
{
    const int mk = G.n_rows;
    arma::vec res = arma::vec(mk, arma::fill::zeros);

    const uvec mid = find(g >= thresh && g <= (1.0 - thresh));
    const uvec big = find(g > (1.0 - thresh));
    const int msize = mid.size();

    const vec mu = sum(Xm.cols(big), 1);
    const vec sig = sum(Xs.cols(big), 1);

    if (msize == 0) 
    {
	res = dt_dm(X, mu, sqrt(sig), G, l);
    } 
    else 
    {
	for (int i = 0; i < pow(2, msize); ++i) 
	{
	    auto b = std::bitset<12>(i);
	    double prod_g = 1.0;

	    vec mu_new = mu;
	    vec sig_new = sig;

	    for (int j = 0; j < msize; ++j) 
	    {
		if ((i >> j) & 1) 
		{
		    mu_new += Xm.col(mid(j));
		    sig_new += Xs.col(mid(j));
		    prod_g *= g(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - g(mid(j)));
		}
	    }
	    res += prod_g * dt_dm(X, mu_new, sqrt(sig_new), G, l);
	}
    }

    return  res;
}


// ------- gradients wrt. s ---------
vec dt_ds(const mat &X, const vec &s, const vec &mu, const vec &sig, 
	const uvec &G, const int l) 
{
    // gradient of tll wrt. s
    const int n = mu.n_rows;
    const int mk = G.n_rows;
    vec res = arma::vec(mk, arma::fill::zeros);

    for (int i = 0; i < n; ++i) 
    {
	double dt_dsig = 0.0;

	dt_dsig = 1.0 / sqrt(2.0 * M_PI) * 
	    (1.0 + mu(i)*mu(i)/(sig(i)*sig(i))) *
	    exp(- 0.5 * mu(i)*mu(i) / (sig(i)*sig(i))) -
	    mu(i)*mu(i)/(sig(i)*sig(i)) * R::dnorm(mu(i) / sig(i), 0, 1, 0);

	for(int j = 1; j <= (2 * l - 1); ++j) 
	{
	    dt_dsig += pow((-1.0), (j-1)) / j * (
		j*j*sig(i) * exp(
		    mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(-mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		) + 
		j*j*sig(i) * exp(
		    -mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::pnorm(mu(i)/sig(i) - j*sig(i), 0, 1, 1, 1)
		) +
		(mu(i)/(sig(i)*sig(i)) - j) * exp(
		    mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::dnorm(-mu(i)/sig(i) - j*sig(i), 0, 1, 1)
		) +
		(-mu(i)/(sig(i)*sig(i)) - j) * exp(
		    -mu(i)*j + 0.5*j*j*sig(i)*sig(i) + 
		    R::dnorm(mu(i)/sig(i) - j*sig(i), 0, 1, 1)
		)
	    );
	}
	
	for (int k = 0; k < mk; ++k) {
	    res(k) += dt_dsig / sig(i) * X(i, G(k)) * X(i, G(k)) * s(G(k));
	}
    }

    return res;
}


vec dell_ds(const mat &X, const mat &Xm, const mat &Xs, const vec &s,
	const vec &g, const uvec &G, const double thresh, const int l) 
{
    const int mk = G.n_rows;
    vec res = arma::vec(mk, arma::fill::zeros);

    const uvec mid = find(g >= thresh && g <= (1.0 - thresh));
    const uvec big = find(g > (1.0 - thresh));
    const int msize = mid.size();

    const vec mu = sum(Xm.cols(big), 1);
    const vec sig = sum(Xs.cols(big), 1);

    if (msize == 0) 
    {
	res = dt_ds(X, s, mu, sqrt(sig), G, l);
    } 
    else 
    {
	double tot = 0.0;
	for (int i = 0; i < pow(2, msize); ++i) 
	{
	    auto b = std::bitset<12>(i);
	    double prod_g = 1.0;

	    vec mu_new = mu;
	    vec sig_new = sig;

	    for (int j = 0; j < msize; ++j) 
	    {
		if ((i >> j) & 1) 
		{
		    mu_new += Xm.col(mid(j));
		    sig_new += Xs.col(mid(j));
		    prod_g *= g(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - g(mid(j)));
		}
	    }
	    res += prod_g * dt_ds(X, s, mu_new, sqrt(sig_new), G, l);
	}
    }

    return  res;
}

// ----------------- mu -------------------
class update_m_fn
{
    public:
	update_m_fn(const vec &y, const mat &X, const vec &m,
		const vec &s, vec g, double lambda, const uword group,
		const uvec G, mat &Xm, const mat &Xs, double thresh, int l) :
	    y(y), X(X), m(m), s(s), g(g), lambda(lambda),
	    group(group), G(G), Xm(Xm), Xs(Xs), thresh(thresh), l(l)
	    { }

	double EvaluateWithGradient(const mat &mG, mat &grad) 
	{ 
	    g(group) = 1;
	    const vec xm = X.cols(G) * mG;

	    // Xm is a reference and is updated on each iteration
	    Xm.col(group) = xm;

	    const double res = ell(Xm, Xs, g, thresh, l) -
		dot(y, xm) +
		lambda * sqrt(sum(s(G) % s(G) + mG % mG));

	    grad = dell_dm(X, Xm, Xs, g, G, thresh, l) -
		X.cols(G).t() * y +
		lambda * mG * pow(dot(s(G), s(G)) + dot(mG, mG), -0.5);

	    return res;
	}

    private:
	const vec &y;
	const mat &X;
	const vec &m;
	const vec &s;
	vec g;
	const double lambda;
	const uword group;
	const uvec G;
	mat &Xm;
	const mat &Xs;
	const double thresh;
	const int l;
};


vec update_m(const vec &y, const mat &X, const vec &m,
	const vec &s, const vec &g, double lambda, uword group,
	const uvec G, mat &Xm, const mat &Xs, const double thresh, const int l)  
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    update_m_fn fn(y, X, m, s, g, lambda, group, G, Xm, Xs, thresh, l);

    arma::vec mG = m(G);
    opt.Optimize(fn, mG);

    return mG;
}



// ----------------- s -------------------
class update_s_fn
{
    public:
	update_s_fn(const vec &y, const mat &X, const vec &m, const vec &s, 
		vec g, const double lambda, const uword group, const uvec G, 
		const mat &Xm, mat &Xs, const double thresh, const int l) :
	    y(y), X(X), m(m), s(s), g(g), lambda(lambda),
	    group(group), G(G), Xm(Xm), Xs(Xs), thresh(thresh), l(l)
	    { }

	double EvaluateWithGradient(const mat &u, mat &grad) 
	{ 
	    // use exp to ensure positive
	    vec sG = exp(u);

	    g(group) = 1;
	    const vec xs = (X.cols(G) % X.cols(G)) * (sG % sG);

	    // update Xs by ref
	    Xs.col(group) = xs;

	    const double res = ell(Xm, Xs, g, thresh, l) -
		accu(log(sG)) +
		lambda * sqrt(dot(sG, sG) + dot(m(G), m(G)));

	    grad = (dell_ds(X, Xm, Xs, s, g, G, thresh, l) -
		1.0 / sG +
		lambda * sG * pow(dot(sG, sG) + dot(m(G), m(G)), -0.5)) % sG;

	    return res;
	}

    private:
	const vec &y;
	const mat &X;
	const vec &m;
	const vec &s;
	vec g;
	const double lambda;
	const uword group;
	const uvec G;
	const mat &Xm;
	mat &Xs;
	const double thresh;
	const int l;
};


vec update_s(const vec &y, const mat &X, const vec &m, const vec &s, vec g, 
	const double lambda, const uword group, const uvec G, 
	const mat &Xm, mat &Xs, const double thresh, const int l)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    update_s_fn fn(y, X, m, s, g, lambda, group, G, Xm, Xs, thresh, l);

    vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


// ----------------- g -------------------
double update_g(const vec &y, const mat &X, const vec &m, const vec &s, 
	vec g, const double lambda, const uword group,
	const uvec &G, const mat &Xm, const mat &Xs,
	const double thresh, const int l, const double w)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    g(group) = 1;
    const double S1 = ell(Xm, Xs, g, thresh, l);

    g(group) = 0;
    const double S0 = ell(Xm, Xs, g, thresh, l);

    const double res =
	log(w / (1- w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * accu(log(2.0 * M_PI * s(G) % s(G))) -
	lambda * sqrt(dot(s(G), s(G)) + dot(m(G), m(G))) -
	S1 + S0 + dot(y, (X.cols(G) * m(G)));
    
    return 1.0 / (1.0 + exp(-res));
}

