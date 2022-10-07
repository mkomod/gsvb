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
    vec ug = vec(ugroups.size());

    for (uword group : ugroups) 
    {
	uvec G = arma::find(groups == group);
	Xm.col(group) = X.cols(G) * mu(G);
	Xs.col(group) = (X.cols(G) % X.cols(G)) * (s(G) % s(G));
	ug(group) = g(G(0));
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
	    // uvec Gc = arma::find(groups != group);
	    
	    mu(G) = update_m(y, X, mu, s, ug, lambda, group, G, Xm, Xs, thresh, l);
	    Xm.col(group) = X.cols(G) * mu(G);

	    s(G)  = update_s(y, X, mu, s, ug, lambda, group, G, Xm, Xs, thresh, l);
	    Xs.col(group) = (X.cols(G) % X.cols(G)) * (s(G) % s(G));

	    double tg = update_g(y, X, mu, s, ug, lambda, group, G, Xm, Xs, 
		    thresh, l, w);
	    for (uword j : G) g(j) = tg;
	    ug(group) = tg;
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


double ell(const mat &Xm, const mat &Xs,  const vec &ug, double thresh, int l)
{
    // computes an upper bound for the expected log-likelihood
    // E_Q [ log(1 + exp(x' b) ] where b ~ Q = SpSL = gN + (1-g) delta_0
    // 
    // The function thresholds values of g to compute the expecation
    //
    const uvec mid = find(ug >= thresh && ug <= (1.0 - thresh));
    const uvec big = find(ug > (1.0 - thresh));

    const int msize = mid.size();
    const int bsize = big.size();
    
    // compute mu and sig
    vec mu = vec(Xs.n_rows, arma::fill::zeros); 
    vec sig = vec(Xs.n_rows, arma::fill::zeros); 

    if (bsize >= 1) {
	mu = sum(Xm.cols(big), 1);
	sig = sum(Xs.cols(big), 1);
    }
    
    double res = 0;

    if (msize == 0 && bsize >= 1)
    {
	res += tll(mu, sqrt(sig), l);
    } 
    else if (msize >= 1 && bsize >= 0) 
    {
	double tot = 0.0;
	for (int i = 0 ? bsize : 1; i < pow(2, msize); ++i) 
	{
	    // auto b = std::bitset<12>(i);
	    double prod_g = 1.0;

	    vec mu_new = mu;
	    vec sig_new = sig;

	    for (int j = 0; j < msize; ++j) 
	    {
		if ((i >> j) & 1) 
		{
		    mu_new += Xm.col(mid(j));
		    sig_new += Xs.col(mid(j));
		    prod_g *= ug(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - ug(mid(j)));
		}
	    }
	    
	    tot += prod_g * tll(mu_new, sqrt(sig_new), l);
	} 

	res += tot;
    } else {
	res = prod((1- ug)) * log(2);
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


vec dell_dm(const mat &X, const mat &Xm, const mat &Xs, const vec &ug,
	const uvec &G, const double thresh, const int l)
{
    const int mk = G.n_rows;
    arma::vec res = arma::vec(mk, arma::fill::zeros);

    const uvec mid = find(ug >= thresh && ug <= (1.0 - thresh));
    const uvec big = find(ug > (1.0 - thresh));
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
		    prod_g *= ug(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - ug(mid(j)));
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
	const vec &ug, const uvec &G, const double thresh, const int l) 
{
    const int mk = G.n_rows;
    vec res = arma::vec(mk, arma::fill::zeros);

    const uvec mid = find(ug >= thresh && ug <= (1.0 - thresh));
    const uvec big = find(ug > (1.0 - thresh));
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
		    prod_g *= ug(mid(j));
		} 
		else 
		{
		    prod_g *= (1 - ug(mid(j)));
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
		const vec &s, vec ug, double lambda, const uword group,
		const uvec G, mat &Xm, const mat &Xs, double thresh, int l) :
	    y(y), X(X), m(m), s(s), ug(ug), lambda(lambda),
	    group(group), G(G), Xm(Xm), Xs(Xs), thresh(thresh), l(l)
	    { }

	double EvaluateWithGradient(const mat &mG, mat &grad) 
	{ 
	    ug(group) = 1;
	    const vec xm = X.cols(G) * mG;

	    // Xm is a reference and is updated on each iteration
	    Xm.col(group) = xm;

	    const double res = ell(Xm, Xs, ug, thresh, l) -
		dot(y, xm) +
		lambda * sqrt(sum(s(G) % s(G) + mG % mG));

	    grad = dell_dm(X, Xm, Xs, ug, G, thresh, l) -
		X.cols(G).t() * y +
		lambda * mG * pow(dot(s(G), s(G)) + dot(mG, mG), -0.5);

	    return res;
	}

    private:
	const vec &y;
	const mat &X;
	const vec &m;
	const vec &s;
	vec ug;
	const double lambda;
	const uword group;
	const uvec G;
	mat &Xm;
	const mat &Xs;
	const double thresh;
	const int l;
};


vec update_m(const vec &y, const mat &X, const vec &m,
	const vec &s, const vec &ug, double lambda, uword group,
	const uvec G, mat &Xm, const mat &Xs, const double thresh, const int l)  
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    update_m_fn fn(y, X, m, s, ug, lambda, group, G, Xm, Xs, thresh, l);

    arma::vec mG = m(G);
    opt.Optimize(fn, mG);

    return mG;
}



// ----------------- s -------------------
class update_s_fn
{
    public:
	update_s_fn(const vec &y, const mat &X, const vec &m, const vec &s, 
		vec ug, const double lambda, const uword group, const uvec G, 
		const mat &Xm, mat &Xs, const double thresh, const int l) :
	    y(y), X(X), m(m), s(s), ug(ug), lambda(lambda),
	    group(group), G(G), Xm(Xm), Xs(Xs), thresh(thresh), l(l)
	    { }

	double EvaluateWithGradient(const mat &u, mat &grad) 
	{ 
	    // use exp to ensure positive
	    vec sG = exp(u);

	    ug(group) = 1;
	    const vec xs = (X.cols(G) % X.cols(G)) * (sG % sG);

	    // update Xs by ref
	    Xs.col(group) = xs;

	    const double res = ell(Xm, Xs, ug, thresh, l) -
		accu(log(sG)) +
		lambda * sqrt(dot(sG, sG) + dot(m(G), m(G)));

	    grad = (dell_ds(X, Xm, Xs, s, ug, G, thresh, l) -
		1.0 / sG +
		lambda * sG * pow(dot(sG, sG) + dot(m(G), m(G)), -0.5)) % sG;

	    return res;
	}

    private:
	const vec &y;
	const mat &X;
	const vec &m;
	const vec &s;
	vec ug;
	const double lambda;
	const uword group;
	const uvec G;
	const mat &Xm;
	mat &Xs;
	const double thresh;
	const int l;
};


vec update_s(const vec &y, const mat &X, const vec &m, const vec &s, vec ug, 
	const double lambda, const uword group, const uvec G, 
	const mat &Xm, mat &Xs, const double thresh, const int l)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    update_s_fn fn(y, X, m, s, ug, lambda, group, G, Xm, Xs, thresh, l);

    vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


// ----------------- g -------------------
double update_g(const vec &y, const mat &X, const vec &m, const vec &s, 
	vec ug, const double lambda, const uword group,
	const uvec &G, const mat &Xm, const mat &Xs,
	const double thresh, const int l, const double w)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    ug(group) = 1;
    const double S1 = ell(Xm, Xs, ug, thresh, l);

    ug(group) = 0;
    const double S0 = ell(Xm, Xs, ug, thresh, l);

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * accu(log(2.0 * M_PI * s(G) % s(G))) -
	lambda * sqrt(dot(s(G), s(G)) + dot(m(G), m(G))) -
	S1 + S0 + dot(y, (X.cols(G) * m(G)));
    
    return 1.0 / (1.0 + exp(-res));
}



// ----------------------------------------
// JENSENS
// updates of mu, s, g with Jensens
// ----------------------------------------

class jen_update_mu_fn
{
    public:
	jen_update_mu_fn(const vec &y, const mat &X, const vec &mu,
		const vec &s, const uvec &G, const double P) :
	    y(y), X(X), mu(mu), s(s), G(G), P(P)
	{};

	double EvaluateWithGradient(const mat &mG, mat &grad)
	{
	    double res = 0;
	    return 0;
	};

    private:
	const vec &y;
	const mat &X;
	const vec &mu;
	const vec &s;
	const uvec &G;
	const double P;
};


vec jen_update_mu()
{
    
}


vec jen_update_s()
{

} 


vec jen_update_g()
{

}




// compute_S <- function(X, m, s, g, groups) 
// {
//     S <- rep(1, nrow(X))

//     for (group in unique(groups)) {
// 	G <- which(groups == group)
// 	S <- S * compute_S_G(X, m, s, g, G)
//     }
//     return(S)
// }


// compute_S_G <- function(X, m, s, g, G)
// {
//     apply(X[ , G], 1, function(x) {
// 	(1 - g[G][1]) + g[G][1] * exp(sum(x * m[G] + 0.5 * x^2 * s[G]^2))
//     })
// }

// n_mgf <- function(X, m, s)
// {
//     apply(X, 1, function(x) {
// 	exp(sum(x * m + 0.5 * x^2 * s^2))
//     })
// }


// opt_mu <- function(m_G, y, X, m, s, g, G, lambda, S) 
// {
//     # maybe combine in a Monte Carlo step rather than use
//     # Jensen's for this part?
//     S <- S * n_mgf(X[ , G], m_G, s[G])

//     sum(log1p(S) - y * (X[ , G] %*% m_G)) +
//     lambda * sqrt(sum(s[G]^2) + sum(m_G^2))
// }



// opt_s <- function(s_G, y, X, m, s, g, G, lambda, S) 
// {
//     S <- S * n_mgf(X[ , G], m[g], s_G)

//     sum(log1p(S)) -
//     sum(log(s_G)) +
//     lambda * sqrt(sum(s_G^2) + sum(m[G]^2))
// }


// opt_g <- function(y, X, m, s, g, G, lambda, S) 
// {
//     mk <- length(G)
//     Ck <- mk * log(2) + (mk -1)/2 * log(pi) + lgamma( (mk + 1) / 2)
//     S1 <- S * n_mgf(X[ , G], m[G], s[G])

//     res <- 
// 	log(w / (1- w)) + 
// 	0.5 * mk - 
// 	Ck +
// 	mk * log(lambda) +
// 	0.5 * sum(log(2 * pi * s[G]^2)) -
// 	lambda * sqrt(sum(s[G]^2) + sum(m[G]^2)) +
// 	sum(y * X[ , G] %*% m[G]) -
// 	sum(log1p(S1)) +
// 	sum(log1p(S))

//     sigmoid(res)
// }




