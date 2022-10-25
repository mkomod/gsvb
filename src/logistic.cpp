#include "logistic.h"
#include <bitset>


// [[Rcpp::export]]
Rcpp::List fit_logistic(vec y, mat X, uvec groups, const double lambda, 
    const double a0, const double b0, vec mu, vec s, vec g, 
    const bool diag_cov, bool track_elbo, const uword track_elbo_every, 
    const uword track_elbo_mcn, const double thresh, const int l, 
    unsigned int niter, unsigned int alg, double tol, bool verbose)
{
    const uword n = X.n_rows;
    const uword p = X.n_cols;
    
    const uvec ugroups = arma::unique(groups);
    const uword M = ugroups.size();
    const double w = a0 / (a0 + b0);
    
    // init new bound
    vec mu_old, s_old, g_old;
    mat Xm = mat(n, M);
    mat Xs = mat(n, M);
    vec ug = vec(M);

    // init jensens
    mat XX = mat(n, p);
    vec P = vec(n , arma::fill::zeros);

    // init jaakkola
    mat XAX = mat(p, p);
    vec jaak_vp = vec(n);
    std::vector<mat> Ss;
	
    // new bound init
    if (alg == 1)
	for (uword group : ugroups) 
	{
	    uword gi = arma::find(ugroups == group).eval().at(0);
	    uvec G = arma::find(groups == group);

	    Xm.col(gi) = X.cols(G) * mu(G);
	    Xs.col(gi) = (X.cols(G) % X.cols(G)) * (s(G) % s(G));
	    ug(gi) = g(G(0));
	}
    
    // jensens init
    if (alg == 2) {
	XX = X % X;
	P = compute_P(X, XX, mu, s, g, groups);
    }

    // jaak init
    if (alg == 3) {
	jaak_vp = jaak_update_l(X, mu, s, g);
	XAX = X.t() * diagmat(a(jaak_vp)) * X;

	// init unristricted covariance matrix
	if (!diag_cov) {
	    for (uword group : ugroups) {
		uvec G = find(groups == group);	
		Ss.push_back(arma::diagmat(s(G)));
	    }
	}
    }

    uword num_iter = niter;
    std::vector<double> elbo_values;
    bool converged = false;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
	mu_old = mu; s_old = s; g_old = g;

	if (alg == 3) {
	    if (diag_cov) {
		jaak_vp = jaak_update_l(X, mu, s, g);
	    } else {
		jaak_vp = jaak_update_l(X, mu, Ss, g, groups, ugroups);
	    }
	    XAX = X.t() * diagmat(a(jaak_vp)) * X;
	}

	// update mu, sigma, gamma
	for (uword group : ugroups)
	{
	    uvec G  = arma::find(groups == group);
	    
	    // update using new bound
	    if (alg == 1)
	    {
		uword gi = arma::find(ugroups == group).eval().at(0);

		mu(G) = nb_update_m(y, X, mu, s, ug, lambda, gi, G, Xm, Xs, thresh, l);
		Xm.col(gi) = X.cols(G) * mu(G);

		s(G)  = nb_update_s(y, X, mu, s, ug, lambda, gi, G, Xm, Xs, thresh, l);
		Xs.col(gi) = (X.cols(G) % X.cols(G)) * (s(G) % s(G));

		double tg = nb_update_g(y, X, mu, s, ug, lambda, gi, G, Xm, Xs, 
			thresh, l, w);
		for (uword j : G) g(j) = tg;
		ug(gi) = tg;
	    }

	    // update using jensens
	    if (alg == 2)
	    {
		P /= compute_P_G(X, XX, mu, s, g, G);

		mu(G) = jen_update_mu(y, X, XX, mu, s, lambda, G, P);
		s(G)  = jen_update_s( y, X, XX, mu, s, lambda, G, P);

		double tg = jen_update_g(y, X, XX, mu, s, lambda, w, G, P);
		for (uword j : G) g(j) = tg;

		P %= compute_P_G(X, XX, mu, s, g, G);
	    }

	    // update using jaakola bound
	    if (alg == 3)
	    {
		uvec Gc = arma::find(groups != group);

		if (diag_cov)
		{
		    mu(G) = jaak_update_mu(y, X, XAX, mu, s(G), g, lambda, G, Gc);
		    s(G)  = jaak_update_s(XAX, mu, s, lambda, G);
		    double tg = jaak_update_g(y, X, XAX, mu, s, g, lambda, w, G, Gc);
		    for (uword j : G) g(j) = tg;
		} 
		else 
		{
		    // get the index of the group
		    uword gi = arma::find(ugroups == group).eval().at(0);
		    mat &S = Ss.at(gi);

		    mu(G) = jaak_update_mu(y, X, XAX, mu, sqrt(diagvec(S)), g, lambda, G, Gc);
		    s(G)  = jaak_update_S(XAX, mu, S, s, lambda, G);

		    double tg = jaak_update_g(y, X, XAX, mu, S, g, lambda, w, G, Gc);
		    for (uword j : G) g(j) = tg;
		}
	    }
	}

	if (track_elbo && (iter % track_elbo_every == 0)) {
	    double e = elbo_logistic(y, X, groups, mu, s, g, Ss, lambda, w, track_elbo_mcn,
		    diag_cov);
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
	double e = elbo_logistic(y, X, groups, mu, s, g, Ss, lambda, w, track_elbo_mcn,
		diag_cov);
	elbo_values.push_back(e);
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = converged,
	Rcpp::Named("iterations") = num_iter,
	Rcpp::Named("S") = Ss,
	Rcpp::Named("elbo") = elbo_values
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
class nb_update_m_fn
{
    public:
	nb_update_m_fn(const vec &y, const mat &X, const vec &m,
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


vec nb_update_m(const vec &y, const mat &X, const vec &m,
	const vec &s, const vec &ug, double lambda, uword group,
	const uvec G, mat &Xm, const mat &Xs, const double thresh, const int l)  
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    nb_update_m_fn fn(y, X, m, s, ug, lambda, group, G, Xm, Xs, thresh, l);

    arma::vec mG = m(G);
    opt.Optimize(fn, mG);

    return mG;
}



// ----------------- s -------------------
class nb_update_s_fn
{
    public:
	nb_update_s_fn(const vec &y, const mat &X, const vec &m, const vec &s, 
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


vec nb_update_s(const vec &y, const mat &X, const vec &m, const vec &s, vec ug, 
	const double lambda, const uword group, const uvec G, 
	const mat &Xm, mat &Xs, const double thresh, const int l)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    nb_update_s_fn fn(y, X, m, s, ug, lambda, group, G, Xm, Xs, thresh, l);
    
    vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


// ----------------- g -------------------
double nb_update_g(const vec &y, const mat &X, const vec &m, const vec &s, 
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
	jen_update_mu_fn(const vec &y, const mat &X, const mat &XX,
		const vec &mu, const vec &s, const double lambda,
		const uvec &G, const vec &P) :
	    y(y), X(X), XX(XX), mu(mu), s(s), lambda(lambda), G(G), P(P)
	{};

	double EvaluateWithGradient(const mat &mG, mat &grad)
	{
	    const vec PP = P % mvnMGF(X.cols(G), XX.cols(G), mG, s(G));

	    double res = accu(log1p(PP) - y % (X.cols(G) * mG)) +
		lambda * sqrt(accu(s(G) % s(G) + mG % mG)); 
	    
	    vec dPPmG = vec(mG.size(), arma::fill::zeros);

	    for (uword j = 0; j < mG.size(); ++j) {
		dPPmG(j) = accu( X.col(G(j)) % PP / (1 + PP) );
	    }

	    grad = dPPmG -
		X.cols(G).t() * y +
		lambda * mG * pow(dot(s(G), s(G)) + dot(mG, mG), -0.5);
	    
	    return res;
	};

    private:
	const vec &y;
	const mat &X;
	const mat &XX;
	const vec &mu;
	const vec &s;
	const double lambda;
	const uvec &G;
	const vec &P;
};


vec jen_update_mu(const vec &y, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const uvec &G, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    jen_update_mu_fn fn(y, X, XX, mu, s, lambda, G, P);

    arma::vec mG = mu(G);
    opt.Optimize(fn, mG);

    return mG;
}


class jen_update_s_fn
{
    public:
	jen_update_s_fn(const vec &y, const mat &X, const mat &XX,
		const vec &mu, const double lambda, const uvec &G, 
		const vec &P) :
	    y(y), X(X), XX(XX), mu(mu), lambda(lambda), G(G), P(P)
	{};

	double EvaluateWithGradient(const mat &u, mat &grad)
	{
	    const vec sG = exp(u);

	    const vec PP = P % mvnMGF(X.cols(G), XX.cols(G), mu(G), sG);

	    double res = accu(log1p(PP)) -
		accu(log(sG)) +
		lambda * sqrt(accu(sG % sG + mu(G) % mu(G)));
	    
	    vec dPPsG = vec(sG.size(), arma::fill::zeros);

	    for (uword j = 0; j < sG.size(); ++j) {
		dPPsG(j) = accu( sG(j) * (X.col(G(j)) % X.col(G(j))) % 
			PP / (1 + PP) );
	    }

	    // df/duG = df/dsG * dsG/du
	    grad = (dPPsG -
		1.0 / sG +
		lambda * sG * pow(dot(sG, sG) + dot(mu(G), mu(G)), -0.5)) % sG;
	    
	    // Rcpp::Rcout << res;
	    return res;
	};

    private:
	const vec &y;
	const mat &X;
	const mat &XX;
	const vec &mu;
	const double lambda;
	const uvec &G;
	const vec &P;
};


vec jen_update_s(const vec &y, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const uvec &G, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    jen_update_s_fn fn(y, X, XX, mu, lambda, G, P);

    arma::vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


double jen_update_g(const vec &y, const mat &X, const mat &XX, const vec &mu,
	const vec &s, const double lambda, const double w, const uvec &G, 
	const vec &P)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    const vec PP = P % mvnMGF(X.cols(G), XX.cols(G), mu(G), s(G));

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * accu(log(2.0 * M_PI * s(G) % s(G))) -
	lambda * sqrt(dot(s(G), s(G)) + dot(mu(G), mu(G))) +
	dot(y, (X.cols(G) * mu(G))) -
	accu(log1p(PP)) + 
	accu(log1p(P));

    return 1.0/(1.0 + exp(-res));
}


// ---------------------------------------- 
// JAAKKOLA
// Updates for mu, s, g, l
// ----------------------------------------
class jaak_update_mu_fn
{
    public:
	jaak_update_mu_fn(const vec &y, const mat &X, const mat &XAX,
		const vec &mu, const vec &sG, const vec &g, const double lambda,
		const uvec &G, const uvec &Gc) :
	    y(y), X(X), XAX(XAX), mu(mu), sG(sG), g(g), lambda(lambda), 
	    G(G), Gc(Gc)
	{};

	double EvaluateWithGradient(const mat &mG, mat &grad)
	{
	    const double res = 0.5 * dot(mG, XAX(G, G) * mG) +
		dot(mG, XAX(G, Gc) * (g(Gc) % mu(Gc))) +
		dot(0.5 - y, X.cols(G) * mG) +
		lambda * sqrt(accu(sG % sG + mG % mG)); 

	    grad = XAX(G, G) * mG +
		XAX(G, Gc) * (g(Gc) % mu(Gc)) +
		X.cols(G).t() * (0.5 - y) +
		lambda * mG * pow(dot(sG, sG) + dot(mG, mG), -0.5);
	    
	    return res;
	};

    private:
	const vec &y;
	const mat &X;
	const mat &XAX;
	const vec &mu;
	const vec &sG;
	const vec &g;
	const double lambda;
	const uvec &G;
	const uvec &Gc;
};


vec jaak_update_mu(const vec &y, const mat &X, const mat &XAX,
	const vec &mu, const vec &s, const vec &g, const double lambda,
	const uvec &G, const uvec &Gc)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    jaak_update_mu_fn fn(y, X, XAX, mu, s, g, lambda, G, Gc);

    vec mG = mu(G);
    opt.Optimize(fn, mG);

    return mG;
}


class jaak_update_s_fn
{
    public:
	jaak_update_s_fn(const mat &XAX, const vec &mu, 
		const double lambda, const uvec &G) :
	    XAX(XAX), mu(mu), lambda(lambda), G(G)
	{};

	double EvaluateWithGradient(const mat &u, mat &grad)
	{
	    const vec sG = exp(u);

	    const double res = 0.5 * accu(diagvec(XAX(G, G)) % sG % sG) -
		accu(log(sG)) +
		lambda * sqrt(accu(sG % sG + mu(G) % mu(G))); 

	    grad = (
		diagvec(XAX(G, G)) % sG -
		1 / sG +
		lambda * sG * pow(dot(sG, sG) + dot(mu(G), mu(G)), -0.5)
	    ) % sG;

	    return res;
	};

    private:
	const mat &XAX;
	const vec &mu;
	const double lambda;
	const uvec &G;
};


vec jaak_update_s(const mat &XAX, const vec &mu, 
	const vec &s, const double lambda, const uvec &G)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 50;
    jaak_update_s_fn fn(XAX, mu,lambda, G);

    vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


double jaak_update_g(const vec &y, const mat &X, const mat &XAX,
	const vec &mu, const vec &s, const vec &g, const double lambda,
	const double w, const uvec &G, const uvec &Gc)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * accu(log(2.0 * M_PI * s(G) % s(G))) -
	lambda * sqrt(dot(s(G), s(G)) + dot(mu(G), mu(G))) +
	dot((y - 0.5), X.cols(G) * mu(G)) -
	0.5 * dot(mu(G), XAX(G, G) * mu(G)) -
	0.5 * accu(diagvec(XAX(G, G)) % s(G) % s(G)) -
	dot(mu(G), XAX(G, Gc) * (g(Gc) % mu(Gc)));

    return 1.0 / (1.0 + exp(-res));
}


double jaak_update_g(const vec &y, const mat &X, const mat &XAX, const vec &mu,
	const mat &S, const vec &g, const double lambda, const double w,
	const uvec &G, const uvec &Gc)
{
    const double mk = G.size();
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    vec ds = diagvec(S);

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * log(det(2.0 * M_PI * S)) -
	lambda * sqrt(sum(ds) + dot(mu(G), mu(G))) +
	dot((y - 0.5), X.cols(G) * mu(G)) -
	0.5 * dot(mu(G), XAX(G, G) * mu(G)) -
	0.5 * accu( XAX(G, G) % S) -
	dot(mu(G), XAX(G, Gc) * (g(Gc) % mu(Gc)));

    return 1.0 / (1.0 + exp(-res));
}


vec jaak_update_l(const mat &X, const vec &mu, const vec &s, const vec &g) 
{
    return sqrt(pow(X * (g % mu), 2) + (X % X) * (g % s % s));
}


vec jaak_update_l(const mat &X, const vec &mu, const std::vector<mat> &Ss,
	const vec &g, const uvec &groups, const uvec &ugroups)
{
    uword n = X.n_rows;
    vec res = pow(X * (g % mu), 2);

    for (uword group : ugroups)
    {
	uword gi = find(ugroups == group).eval().at(0);
	uvec G = find(groups == group);

	mat S = Ss.at(gi);
	res += arma::sum(g(gi) * (X.cols(G) * S) % X.cols(G), 1);
    }

    return sqrt(res);
}


vec a(const vec &x)
{
    return (sigmoid(x) - 0.5) / x;
}


// ---------------------------------------- 
// JAAKKOLA
// Updates for S
// ----------------------------------------
class jaak_update_S_fn
{
    public:
	jaak_update_S_fn(const mat &XAX, const vec &mu, const double lambda, 
		const uvec &G) :
	    XAX(XAX), mu(mu), lambda(lambda), G(G) { }

	double EvaluateWithGradient(const arma::mat &w, arma::mat &grad) {

	    const mat psi = XAX(G, G);
	    const mat S = arma::inv(psi + arma::diagmat(w));
	    const vec ds = arma::diagvec(S);

	    const double res = 0.5 * arma::trace(psi * S) -
		0.5 * log(arma::det(S)) + 
		lambda * pow(sum(ds) + dot(mu(G), mu(G)), 0.5);

	    // gradient wrt. w
	    double tw = 0.5 * lambda * pow(sum(ds) + dot(mu(G), mu(G)), -0.5);
	    grad = 0.5 * (S % S) * (w - 2.0 * tw);

	    return res;
	}

    private:
	const mat &XAX;
	const vec &mu;
	const double lambda;
	const uvec &G;
};


vec jaak_update_S(const mat &XAX, const vec &mu, mat &S, const vec &s, 
	const double lambda, const uvec &G)
{
    ens::L_BFGS opt;
    jaak_update_S_fn fn(XAX, mu, lambda, G);
    opt.MaxIterations() = 8;
    
    vec sG = s(G);
    opt.Optimize(fn, sG);

    // update S
    S = arma::inv(XAX(G, G) + diagmat(sG)); 
    return sG;
}


// ---------------------------------------- 
// ELBO
// ----------------------------------------
// [[Rcpp::export]]
double elbo_logistic(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const vec &s, const vec &g, const std::vector<mat> &Ss,
	const double lambda, const double w, const uword mcn, const bool diag)
{
    double res = 0.0;

    uvec ugroups = arma::unique(groups);

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
	    vec beta_G = vec(mk, arma::fill::zeros);
	    if (diag) {
		beta_G = arma::randn(mk) % s(G) + mu(G);
	    } else {
		uword gi = arma::find(ugroups == group).eval().at(0);
		mat S = Ss.at(gi);
		
		beta_G = arma::sqrtmat_sympd(S) * arma::randn(mk) + mu(G);
	    }

	    mci -= lambda * g(k) * norm(beta_G, 2);
	    
	    if (R::runif(0, 1) <= g(k)) {
		beta(G) = beta_G;
	    }
	}
	
	uvec nzero = find(beta != 0);
	vec Xb = X.cols(nzero) * beta(nzero);
	mci += dot(y, Xb) - accu(Xb.for_each(log1pexp));
    }
    mci = mci / static_cast<double>(mcn);
    res += mci;

    return(res);
}
