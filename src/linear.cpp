#include "linear.h"


// [[Rcpp::export]]
Rcpp::List fit_linear(vec y, mat X, uvec groups, const double lambda, const double a0,
    const double b0, const double tau_a0, const double tau_b0, vec mu, vec s, 
    vec g, bool diag_cov, bool track_elbo, const uword track_elbo_every, 
    const uword track_elbo_mcn, unsigned int niter, double tol, bool verbose, 
	const uword ordering)
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
	uvec g_order = ugroups;

    // if not constrained we are using a full covariance for S
    std::vector<mat> Ss;
    if (!diag_cov) {
		for (uword group : ugroups) {
			uvec G = find(groups == group);	
			Ss.push_back(arma::diagmat(s(G)));
		}
    }
    vec v = vec(ugroups.size(), arma::fill::ones);

    vec mu_old, s_old, g_old, v_old;
    double tau_a = tau_a0, tau_b = tau_b0, e_tau = tau_a0 / tau_b0;

    uword num_iter = niter;
    bool converged = false;
    std::vector<double> elbo_values;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
		mu_old = mu; g_old = g;
		if (diag_cov) {
			s_old = s; 
		} else {
			v_old = v;
		}

		// order the groups based
		if (ordering == 1) 
		{
			// Rcpp::Rcout << "Order by rand\n";
			// random ordering
			g_order = arma::shuffle(ugroups);
		} 
		else if (ordering == 2) 
		{
			// Rcpp::Rcout << "Order by mag\n";
			// sort by magnitude of mu
			vec beta_mag = vec(ugroups.size(), arma::fill::zeros);
			for (uword i = 0; i < ugroups.size(); ++i) 
			{
				uvec G = find(groups == ugroups(i));
				beta_mag(i) = arma::norm(mu(G), 2);
			}
			g_order = ugroups(sort_index(beta_mag, "descend"));
		}	


		// update expected value of tau^2
		e_tau = tau_a / tau_b;

		// update mu, sigma, gamma
		for (uword group : g_order)
		{
			uvec G  = arma::find(groups == group);
			uvec Gc = arma::find(groups != group);
			
			if (diag_cov)
			{
				mu(G) = update_mu(G, Gc, xtx, yx, mu, s(G), g, e_tau, lambda);
				s(G)  = update_s(G, xtx, mu, s, e_tau, lambda);
				double tg = update_g(G, Gc, xtx, yx, mu, s, g, e_tau, lambda, w);
				for (uword j : G) g(j) = tg;
			} 
			else 
			{
				// get the index of the group
				uword gi = arma::find(ugroups == group).eval().at(0);
				mat &S = Ss.at(gi);

				mu(G) = update_mu(G, Gc, xtx, yx, mu, sqrt(diagvec(S)), g, e_tau, lambda);
				v(gi)  = update_S(G, xtx, mu, S, v(gi), e_tau, lambda);
				double tg = update_g(G, Gc, xtx, yx, mu, S, g, e_tau, lambda, w);
				for (uword j : G) g(j) = tg;
			}
		}
		
		// update tau_a, tau_b
		double R = diag_cov ?
			compute_R(yty, yx, xtx, groups, mu, s, g, p, false) :
			compute_R(yty, yx, xtx, groups, mu, Ss, g, p, false);

		update_a_b(tau_a, tau_b, tau_a0, tau_b0, R, n);

		// check for break, print iter
		Rcpp::checkUserInterrupt();
		if (verbose) Rcpp::Rcout << iter;
		
		// compute the ELBO if option enabled
		if (track_elbo && (iter % track_elbo_every == 0)) {
			double e = diag_cov ?
			elbo_linear_c(yty, yx, xtx, groups, n, p, mu, s, g, tau_a, tau_b,
				lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false) :
			elbo_linear_u(yty, yx, xtx, groups, n, p, mu, Ss, g, tau_a, tau_b,
				lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false);

			elbo_values.push_back(e);
		}

		// check convergence
		bool var_conv = diag_cov ? sum(abs(s_old - s)) < tol : sum(abs(v_old - v)) < tol; 

		if (sum(abs(mu_old - mu)) < tol &&
			var_conv &&
			sum(abs(g_old - g)) < tol) 
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
		double e = diag_cov ?
			elbo_linear_c(yty, yx, xtx, groups, n, p, mu, s, g, tau_a, tau_b,
			lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false) :
			elbo_linear_u(yty, yx, xtx, groups, n, p, mu, Ss, g, tau_a, tau_b,
			lambda, a0, b0, tau_a0, tau_b0, track_elbo_mcn, false);
		elbo_values.push_back(e);
    }
    
    return Rcpp::List::create(
		Rcpp::Named("mu") = mu,
		Rcpp::Named("sigma") = s,
		Rcpp::Named("S") = Ss,
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


// Update mu using monte carlo integration to estimate the intractable integral 
// the function is slower than update_mu_fn and gives similar results.
// This function is not used within the main
//
// double update_mu_fn_2(const vec &m, const mat &xtx, const vec &yx, const vec &mu, 
// 	const vec &s, const vec &g, const double sigma, const double lambda, 
// 	const uvec &G, const uvec &Gc, const uword mcn)
// {
//     const double sigma_s = pow(sigma, -2.0);
//     double mci = 0.0;
//     for (uword iter = 0; iter < mcn; ++iter) {
// 	mci += norm(arma::randn(size(m)) % s(G) + m, 2);
//     }
//     mci = mci / static_cast<double>(mcn);

//     const double res = 0.5 * sigma_s * dot(m.t() * xtx(G, G), m) + 
// 	sigma_s * dot(m.t() * xtx(G, Gc), (g(Gc) % mu(Gc))) -
// 	sigma_s * dot(yx(G), m) +
// 	lambda * mci;

//     return res;
// }


// ----------------- sigma -------------------
class update_s_fn
{
    public:
	update_s_fn(const uvec &G, const mat &xtx, const vec &mu, 
		const double e_tau, const double lambda) :
	    G(G), xtx(xtx), mu(mu), e_tau(e_tau), lambda(lambda) { }

	double EvaluateWithGradient(const arma::mat &u, arma::mat &grad) {
	    mat s = exp(u); // we need to force s to be positive everywhere

	    const double res = 0.5 * e_tau * dot(diagvec(xtx(G, G)), s % s) -
		accu(log(s)) + lambda * pow(dot(s, s) + dot(mu(G), mu(G)), 0.5);

	    // since we're optimzing over u, we need to return the gradient with
	    // respect to u. By the chain rule the grad is:
	    // 
	    // d / du = d / ds * ds / du
	    grad = (e_tau * diagvec(xtx(G, G)) % s -
		1/s + lambda * s * pow(dot(s, s) + dot(mu(G), mu(G)), -0.5)) % s;

	    return res;
	}

    private:
	const uvec &G;
	const mat &xtx;
	const vec &mu;
	const double e_tau;
	const double lambda;
};


vec update_s(const uvec &G, const mat &xtx, const vec &mu, 
	const vec &s, const double e_tau, const double lambda)
{
    ens::L_BFGS opt;
    update_s_fn fn(G, xtx, mu, e_tau, lambda);
    opt.MaxIterations() = 8;
    
    // we are using the relationship s = exp(u) to
    // for s to be positive everywhere
    vec u = log(s(G));
    opt.Optimize(fn, u);

    return exp(u);
}


// ----------------- S -------------------
class update_S_fn
{
    public:
	update_S_fn(const uvec &G, const mat &xtx, const vec &mu, 
		const double e_tau, const double lambda) :
	    G(G), xtx(xtx), mu(mu), e_tau(e_tau), lambda(lambda) { }

	double EvaluateWithGradient(const mat &v, mat &grad) {
	    const mat psi = xtx(G, G);
	    const vec w = vec(G.size(), arma::fill::value(v(0, 0)));
	    const mat S = arma::inv(e_tau * psi + arma::diagmat(w));
	    const vec ds = arma::diagvec(S);

	    const double res = 0.5 * e_tau * arma::trace(psi * S) -
		0.5 * log(arma::det(S)) + 
		lambda * pow(sum(ds) + dot(mu(G), mu(G)), 0.5);

	    // gradient wrt. v
	    double tv = 0.5 * lambda * pow(sum(ds) + dot(mu(G), mu(G)), -0.5);
	    // grad = 0.5 * (v(0, 0) - 2.0 * tv) * accu(S % S);
	    grad = (0.5 * v(0, 0) - tv) * accu(S % S);

	    return res;
	}

    private:
	const uvec &G;
	const mat &xtx;
	const vec &mu;
	const double e_tau;
	const double lambda;
};


double update_S(const uvec &G, const mat &xtx, const vec &mu, 
	mat &S, double s, const double e_tau, const double lambda)
{
    ens::L_BFGS opt;
    update_S_fn fn(G, xtx, mu, e_tau, lambda);
    opt.MaxIterations() = 8;
   
    mat v = mat(1, 1);
    v(0, 0) = s;
    opt.Optimize(fn, v);

    // update S
    S = arma::inv(e_tau * xtx(G, G) + v(0, 0) * arma::eye(G.size(), G.size())); 
    return v(0, 0);
}


// ----------------- gamma -------------------
double update_g(const uvec &G, const uvec &Gc, const mat &xtx,
	const vec &yx, const vec &mu, const vec &s, const vec &g, double e_tau,
	double lambda, double w)
{
    const double mk = G.size();
    double res = log(w / (1.0 - w)) + 0.5*mk + e_tau * arma::dot(yx(G), mu(G)) +
	0.5 * mk * log(2.0 * M_PI) +
	sum(log(s(G))) -
	mk * log(2.0) - 0.5 * (mk - 1.0) * log(M_PI) - lgamma(0.5 * (mk + 1)) +
	mk * log(lambda) - 
	lambda * sqrt(sum(pow(s(G), 2.0)) + sum(pow(mu(G), 2.0))) -
	0.5 * e_tau * dot(diagvec(xtx(G, G)), pow(s(G), 2.0)) -
	0.5 * e_tau * dot(mu(G).t() * xtx(G, G), mu(G)) -
	e_tau * dot(mu(G).t() * xtx(G, Gc), g(Gc) % mu(Gc));

    return sigmoid(res);
}


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
	lambda * sqrt(sum(diag_S) + sum(mu(G) % mu(G))) -
	0.5 * e_tau * accu(xtx(G, G) % S) -
	0.5 * e_tau * dot(mu(G).t() * xtx(G, G), mu(G)) -
	e_tau * dot(mu(G).t() * xtx(G, Gc), g(Gc) % mu(Gc));

    return sigmoid(res);
}


// ----------------- tau ---------------------
// Used for testing and not directly used within the C++
// implementation.
double update_a_b_obj(const double ta, const double tb, const double ta0,
	const double tb0, const double R, const double n) 
{
    double res = ta * log(tb) - R::lgammafn(ta) +
	(0.5 * n + ta0 - ta) * (log(tb) - R::digamma(ta)) +
	(0.5 * R + tb0 - tb) * (ta / tb);
    return res;
}


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

    // Note: we restrict tau:a, tau:b to be strictly positive
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
//
// TEST WRITTEN: [sort of]
// [[Rcpp::export]]
double elbo_linear_c(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const vec &s, const vec &g,
	const double tau_a, const double tau_b, const double lambda, 
	const double a0, const double b0, const double tau_a0, 
	const double tau_b0, const uword mcn, const bool approx, 
	const double approx_thresh)
{
    const double w = a0 / (a0 + b0);
    const double e_tau = tau_a / tau_b;

    double res = 0.0;
    const double R = compute_R(yty, yx, xtx, groups, mu, s, g, p, 
	    approx, approx_thresh);

    res += -0.5 * n * log(2 * M_PI) - 
	0.5 * n * (log(tau_b) + R::digamma(tau_a)) -
	0.5 * e_tau * yty -			// yty := <y, y>
	0.5 * e_tau * R +			// S := (X'X)_ij E[b_i b_j]
	e_tau * dot(yx, g % mu) +		// yx := X'y
	0.5 * sum(g % log(2 * M_PI * pow(s, 2.0)));
    
    // compute the terms that depend on gamma_k
    for (uword K : unique(groups).eval()) {
	uvec G = find(groups == K);
	uword k = G(0);
	double mk = G.size();		// mk = group size
	
	// Ck: part of the normalization constant for the Multivariate
	// double Exp dist
	double Ck = -mk*log(2.0) - 0.5*(mk-1.0)*log(M_PI) - lgamma(0.5*(mk+1));

	res += g(k) * Ck +
	    0.5 * g(k) * mk +
	    g(k) * mk * log(lambda) -
	    g(k) * log((1e-8 + g(k)) / (1e-8 + w)) -	// add 1e-8 to prevent -Inf
	    (1 - g(k)) * log((1-g(k) + 1e-8) / (1 - w));
	
	// Compute the Monte-Carlo integral of E_Q [ lambda * || b_{G_k} || ]
	double mci = 0.0;
	for (uword iter = 0; iter < mcn; ++iter) {
	    mci += norm(arma::randn(mk) % s(G) + mu(G), 2);
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

// un-constrained
// [[Rcpp::export]]
double elbo_linear_u(const double yty, const vec &yx, const mat &xtx, const uvec &groups,
	const uword n, const uword p, const vec &mu, const std::vector<mat> &Ss, 
	const vec &g, const double tau_a, const double tau_b, const double lambda,
	const double a0, const double b0, const double tau_a0, const double tau_b0, 
	const uword mcn, const bool approx, const double approx_thresh)
{
    const uvec ugroups = unique(groups);
    const double w = a0 / (a0 + b0);
    const double e_tau = tau_a / tau_b;

    double res = 0.0;
    const double R = compute_R(yty, yx, xtx, groups, mu, Ss, g, p, 
	    approx, approx_thresh);

    res += -0.5 * n * log(2 * M_PI) - 
	0.5 * n * (log(tau_b) + R::digamma(tau_a)) -
	0.5 * e_tau * yty -			// yty := <y, y>
	0.5 * e_tau * R +			// S := (X'X)_ij E[b_i b_j]
	e_tau * dot(yx, g % mu);		// yx := X'y

    // compute the terms that depend on gamma_k
    for (uword group : ugroups) 
    {
	uvec G = find(groups == group);	// indices of group members 
	uword k = G(0);
	double mk = G.size();
	
	// get index of group
	uword group_index = find(ugroups == group).eval().at(0);
	mat S = Ss.at(group_index);
	
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


// ------------ compute R -------------
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

    const uvec ugroups = unique(groups);
    double xtx_bi_bj = 0.0;

    for (uword i : indx) {
	uword group_i = groups(i);
	uword gi_indx = find(group_i == ugroups).eval().at(0);
	uword min_indx = min(find(groups == group_i));

	for (uword j : indx) {
	    uword group_j = groups(j);

	    if (group_i == group_j) {
		double S_ij = Ss.at(gi_indx)(i - min_indx, j - min_indx);
		xtx_bi_bj += (xtx(i, j) * g(i) * (S_ij + mu(i) * mu(j)));
	    } else {
		xtx_bi_bj += (xtx(i, j) * g(i) * g(j) * mu(i) * mu(j));
	    }
	}
    }

    double R = yty + xtx_bi_bj - 2.0 * dot(yx, g % mu);
    return R;
}


double compute_R(const double yty, const vec &yx, const mat &xtx, 
	const uvec &groups, const vec &mu, const vec &s, const vec &g, 
	const uword p, const bool approx, const double approx_thresh) 
{
    // indices to sum over, if approx, then indx = g >= threshold
    const uvec indx = approx ? 
	find(g >= approx_thresh) : arma::regspace<uvec>(0, p-1);

    double xtx_bi_bj = 0.0;

    for (uword i : indx) {
	uword group_i = groups(i);

	for (uword j : indx) {
	    uword group_j = groups(j);

	    if (i == j) {
		xtx_bi_bj += (xtx(i, i) * g(i) * (s(i) * s(i) + mu(i) * mu(i)));
	    }
	    if ((i != j) && (group_i == group_j)) {
		xtx_bi_bj += (xtx(i, j) * g(i) * mu(i) * mu(j));
	    }
	    if ((i != j) && (group_i != group_j)) {
		xtx_bi_bj += (xtx(i, j) * g(i) * g(j) * mu(i) * mu(j));
	    }
	}
    }

    double R = yty + xtx_bi_bj - 2.0 * dot(yx, g % mu);
    return R;
}
