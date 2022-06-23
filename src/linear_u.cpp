#include "linear_u.h"


// [[Rcpp::export]]
Rcpp::List fit_linear_u(vec y, mat X, uvec groups, const double lambda, const double a0,
    const double b0, const double tau_a0, const double tau_b0, vec mu, vec s, 
    vec g, bool track_elbo, const uword track_elbo_every, const unsigned int 
    track_elbo_mcn, unsigned int niter, double tol, bool verbose)
{
    const uword n = X.n_rows;
    const uword p = X.n_cols;
    const double w = a0 / (a0 + b0);
    
    // compute commonly used expressions
    const mat xtx = X.t() * X;
    const double yty = dot(y, y);
    const vec yx = (y.t() * X).t();
    
    // init
    const uvec ugroups = arma::unique(groups);

    std::vector<mat> Ss;
    for (uword group : ugroups) {
	uvec G = find(groups == group);	
	Ss.push_back(arma::diagmat(s(G)));
    }

    vec mu_old, s_old, g_old;
    double tau_a = tau_a0, tau_b = tau_b0, e_tau = tau_a0 / tau_b0;

    uword num_iter = niter;
    bool converged = false;
    std::vector<double> elbo_values;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
	mu_old = mu; s_old = s; g_old = g;

	// update expected value of 1/tau^2
	e_tau = tau_a / tau_b;

	// update mu, sigma, gamma
	for (uword group : ugroups)
	{
	    uvec G  = arma::find(groups == group);
	    uvec Gc = arma::find(groups != group);
	    
	    mat &S = Ss.at(group);

	    mu(G) = update_mu(G, Gc, xtx, yx, mu, diagvec(S), g, e_tau, lambda);
	    s(G)  = update_S(G, xtx, mu, S, s(G), e_tau, lambda);
	    double tg = update_g(G, Gc, xtx, yx, mu, S, g, e_tau, lambda, w);
	    for (uword j : G) g(j) = tg;
	}

	// update tau_a, tau_b
	double R = compute_R(yty, yx, xtx, groups, mu, Ss, g, p, false);
	update_a_b(tau_a, tau_b, tau_a0, tau_b0, R, n);

	// check for break, print iter
	Rcpp::checkUserInterrupt();
	if (verbose) Rcpp::Rcout << iter;
	
	// compute the ELBO if option enabled
	if (track_elbo && (iter % track_elbo_every == 0)) {
	    double e = elbo(yty, yx, xtx, groups, n, p, mu, Ss, g, tau_a, tau_b,
		    lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false);
	    elbo_values.push_back(e);
	}

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
    if (track_elbo) {
	double e = elbo(yty, yx, xtx, groups, n, p, mu, Ss, g, tau_a, tau_b,
		lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false);
	elbo_values.push_back(e);
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = Ss,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("tau_a") = tau_a,
	Rcpp::Named("tau_b") = tau_b,
	Rcpp::Named("converged") = converged,
	Rcpp::Named("iterations") = num_iter,
	Rcpp::Named("elbo") = elbo_values
    );
}


// ----------------- mu -------------------
class update_mu_fn
{
    public:
	update_mu_fn(const uvec &G, const uvec &Gc, const mat &xtx, 
		const vec &yx, const vec &mu, const vec &s, const vec &g, 
		const double e_tau, const double lambda) :
	    G(G), Gc(Gc), xtx(xtx), yx(yx), mu(mu), s(s), g(g), 
	    e_tau(e_tau), lambda(lambda)
	    { }

	double EvaluateWithGradient(const arma::mat &m, arma::mat &grad) {

	    const double res = 0.5 * e_tau * dot(m.t() * xtx(G, G), m) + 
		e_tau * dot(m.t() * xtx(G, Gc), (g(Gc) % mu(Gc))) -
		e_tau * dot(yx(G), m) +
		lambda * pow(dot(s, s) + dot(m, m), 0.5);

	    grad = e_tau * xtx(G, G) * m + 
		e_tau * xtx(G, Gc) * (g(Gc) % mu(Gc)) -
		e_tau * yx(G) +
		lambda * m * pow(dot(s, s) + dot(m, m), -0.5);

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
	const double e_tau;
	const double lambda;
};


// [[Rcpp::export]]
vec update_mu(const uvec &G, const uvec &Gc, const mat &xtx, 
	const vec &yx, const vec &mu, const vec &s, const vec &g, 
	const double e_tau, const double lambda)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = 8;
    update_mu_fn fn(G, Gc, xtx, yx, mu, s, g, e_tau, lambda);

    vec m = mu(G);
    opt.Optimize(fn, m);

    return m;
}


// ----------------- sigma -------------------
class update_S_fn
{
    public:
	update_S_fn(const uvec &G, const mat &xtx, const vec &mu, 
		const double e_tau, const double lambda) :
	    G(G), xtx(xtx), mu(mu), e_tau(e_tau), lambda(lambda) { }

	double EvaluateWithGradient(const arma::mat &w, arma::mat &grad) {
	    const mat psi = xtx(G, G);
	    const mat S = arma::inv(e_tau * psi + arma::diagmat(w));
	    const vec ds = arma::diagvec(S);

	    const double res = 0.5 * e_tau * arma::trace(psi * S) -
		0.5 * log(arma::det(S)) + 
		lambda * pow(dot(ds, ds) + dot(mu(G), mu(G)), 0.5);

	    // gradient wrt. w
	    double tw = 0.5 * lambda * pow(sum(ds) + dot(mu(G), mu(G)), -0.5);
	    grad = 0.5 * (S % S) * (w - 2.0 * tw);

	    return res;
	}

    private:
	const uvec &G;
	const mat &xtx;
	const vec &mu;
	const double e_tau;
	const double lambda;
};


vec update_S(const uvec &G, const mat &xtx, const vec &mu, 
	mat &S, vec s, const double e_tau, const double lambda)
{
    ens::L_BFGS opt;
    update_S_fn fn(G, xtx, mu, e_tau, lambda);
    opt.MaxIterations() = 8;
    
    // we are using the relationship s = exp(u) to
    // for s to be positive everywhere
    opt.Optimize(fn, s);

    // update S
    S = arma::inv(e_tau * xtx(G, G) + diagmat(s)); 
    return s;
}


// ----------------- gamma -------------------
double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const mat &S, const vec &g, double e_tau,
	double lambda, double w)
{
    const double mk = G.size();
    vec diag_S = diagvec(S);
    double res = log(w / (1.0 - w)) + 0.5*mk + e_tau * arma::dot(yx(G), mu(G)) +
	0.5 * log(det(2.0 * M_PI * S)) +
	mk * log(2.0) - 0.5 * (mk - 1.0) * log(M_PI) - lgamma(0.5 * (mk + 1)) + // log(Ck)
	mk * log(lambda) - 
	lambda * sqrt(sum(diag_S % diag_S) + sum(mu(G) % mu(G))) -
	0.5 * e_tau * accu(xtx(G, G) % S) -
	0.5 * e_tau * dot(mu(G).t() * xtx(G, G), mu(G)) -
	e_tau * dot(mu(G).t() * xtx(G, Gc), g(Gc) % mu(Gc));

    return sigmoid(res);
}


// ----------------- tau ---------------------
class update_a_b_fn
{
    public:
	update_a_b_fn(const double ta0, const double tb0, const double R,
		const double n) :
	    ta0(ta0), tb0(tb0), R(R), n(n) { }

	double EvaluateWithGradient(const arma::mat &pars, arma::mat &grad) 
	{
	    // force tau_a to be positive 
	    const double ta = exp(pars(0, 0));	// we need to restrict ta to be positive
	    const double tb = exp(pars(1, 0));

	    const double res = ta * log(tb) - R::lgammafn(ta) +
		(0.5 * n + ta0 - ta) * (log(tb) - R::digamma(ta)) +
		(0.5 * R + tb0 - tb) * (ta / tb);
	    
	    // gradient of res with respect to tau:a
	    // by the chain rule dfdu = df/da * da/du
	    const double dfdu = (log(tb) - R::digamma(ta) -
		(log(tb) - R::digamma(ta)) -
		(0.5 * n + ta0 - ta) * R::trigamma(ta) +
		(0.5 * R + tb0 - tb) * (1.0 / tb)) * ta;

	    // gradient of res with respect to tau:b
	    // by the chain rule dfdw = df/db * db/dw
	    const double dfdw =  (ta / tb +
		(0.5 * n + ta0 - ta) / tb -
		(0.5 * R + tb0 - tb) * (ta / (tb * tb)) -
		(ta / tb)) * tb;
	    
	    // save the grad
	    grad(0, 0) = dfdu;
	    grad(1, 0) = dfdw;

	    return res;
	}

    private:
	const double ta0;
	const double tb0;
	const double R;
	const double n;
};


void update_a_b(double &tau_a, double &tau_b, const double tau_a0,
	const double tau_b0, const double R, const double n)
{
    ens::L_BFGS opt(50, 1000); // (numBasis, maxIterations)
    update_a_b_fn fn(tau_a0, tau_b0, R, n);
    
    mat pars = mat(2, 1, arma::fill::zeros);

    // Note: we must restrict tau:a, tau:b to be positive
    // so we let, tau:a = exp(u), tau:b = exp(w)
    // optimization is then done over u and w, and then transformed back 
    // to tau:a and tau:b
    pars(0, 0) = log(tau_a);
    pars(1, 0) = log(tau_b);

    opt.Optimize(fn, pars);
    
    // update tau_a and tau_b
    tau_a = exp(pars(0, 0));
    tau_b = exp(pars(1, 0));
}


// --------------- ELBO ----------------
//
// Compute the evidence lower bound (ELBO), used to assess the model fit.
// The ELBO has also been used as a convergence diagnostic.
//
// ELBO := E_Q [ log L(D; b) + log Q / Pi ] <= log Pi_D
// where Q: variational family, Pi: prior, Pi_D: model evidence
// l(D; beta): likelihood

// [[Rcpp::export]]
double elbo(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const double tau_a, const double tau_b, const double lambda,
	const double a0, const double b0, const double tau_a0, const double tau_b0, 
	const uword mcn, const bool approx, const double approx_thresh)
{
    const double w = a0 / (a0 + b0);
    const double e_tau = tau_a / tau_b;

    double res = 0.0;
    const double R = compute_R(yty, yx, xtx, groups, mu, Ss, g, p, approx, approx_thresh);

    res += -0.5 * n * log(2 * M_PI) - 
	0.5 * n * (log(tau_b) + R::digamma(tau_a)) -
	0.5 * e_tau * yty -			// yty := <y, y>
	0.5 * e_tau * R +			// S := (X'X)_ij E[b_i b_j]
	e_tau * dot(yx, g % mu);		// yx := X'y

    
    // compute the terms that depend on gamma_k
    for (uword group : unique(groups).eval()) 
    {
	uvec G = find(groups == group);	// indices of group members 
	uword k = G(0);
	double mk = G.size();
	mat S = Ss.at(group);
	
	// Normalization const, Ck: double exp, Sk: multivariate norm
	double Ck = -mk*log(2.0) - 0.5*(mk-1.0)*log(M_PI) - lgamma(0.5*(mk+1));
	double Sk = 0.5 * log(arma::det(2 * M_PI * S));

	res += g(k) * Sk + 
	    g(k) * Ck +
	    0.5 * g(k) * mk +
	    g(k) * mk * log(lambda) -
	    g(k) * log((1e-8 + g(k)) / (1e-8 + w)) -	// add 1e-8 to prevent -Inf
	    (1 - g(k)) * log((1-g(k) + 1e-8) / (1 - w));
	
	// Compute the Monte-Carlo integral of E_Q [ lambda * || b_{G_k} || ]
	double mci = 0.0;
	for (uword iter = 0; iter < mcn; ++iter) {
	    // X ~ N(0, I)
	    // Y = S^1/2 X + mu => Y ~ N(mu, S)
	    mci += norm(sqrtmat(S) * arma::randn(mk) + mu(G), 2);
	}
	mci = mci / static_cast<double>(mcn);

	res -= lambda * g(k) * mci;
    }

    // compute the expected value of E_G^-1 [ log dG^-1(a', b') / dG^-1(a, b)]
    res += tau_a * log(tau_b) - tau_a0 * log(tau_b0) + R::lgammafn(tau_a0)
	- R::lgammafn(tau_a) + (tau_a0 - tau_a)*(log(tau_b) + R::digamma(tau_a)) +
	(tau_b0 - tau_b) * tau_a / tau_b;

    return(res);
}


// ------------ compute expected (sqrd) residual -------------
//
// R := E [ | y - Xb |^2 ]		(the expected residuals)
//    = <y, y> - 2 <yx, g o mu> + S_i S_j xtx_ij E[b_i b_j]
// where S_i: sum over i, o: elementwise product
//
// Used in the ELBO and in the opt of tau:a, taub
double compute_R(const double yty, const vec &yx, const mat &xtx,
	const uvec &groups, const vec &mu, const std::vector<mat> &Ss,
	const vec &g, const uword p, const bool approx, 
	const double approx_thresh) 
{
    // indices to sum over, if approx, then indx = g >= threshold
    const uvec indx = approx ? 
	find(g >= approx_thresh) : arma::regspace<uvec>(0, p-1);

    double xtx_bi_bj = 0.0;

    for (uword i : indx) {
	uword group_i = groups(i);
	uword min_indx = min(find(groups == group_i));

	for (uword j : indx) {
	    uword group_j = groups(j);

	    if (group_i == group_j) {
		double S_ij = Ss.at(group_i)(i - min_indx, j - min_indx);
		xtx_bi_bj += (xtx(i, j) * g(i) * (S_ij + mu(i) * mu(j)));
	    } else {
		xtx_bi_bj += (xtx(i, j) * g(i) * g(j) * mu(i) * mu(j));
	    }
	}
    }

    double R = yty + xtx_bi_bj - 2.0 * dot(yx, g % mu);
    return R;
}
