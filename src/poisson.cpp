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
    const vec yX = X.t() * y;
    
    // init
    vec mu_old, s_old, g_old;
    mat XX;
    vec P;
    s_old = s; // init s_old

    // only used for diag_cov = FALSE
    std::vector<mat> Ss;
    std::vector<mat> Us;

    if (diag_cov) {
	XX = X % X;
	P = compute_P(X, XX, mu, s, g, groups);
    } else {

	// populate the covariance matrices and chol decompositions
	for (uword i = 0; i < ugroups.size(); ++i) {
	    uvec G  = arma::find(groups == ugroups(i));

	    mat S = arma::diagmat(s(G) % s(G));

	    if (S.n_cols != 1) {
		uvec upper_indices = trimatu_ind( size(S), 1 );
		uvec lower_indices = trimatl_ind( size(S), -1 );
		
		// const arma::mat X_act = X.cols(G);
		// const arma::vec b_act = mu.rows(G);
		// const arma::vec Xb_act = X_act * b_act;
		// const arma::mat W = diagmat(exp(Xb_act));
		// const arma::mat Omega = (X_act.t() * W * X_act + 0.01 * arma::eye(G.size(), G.size())).i();
		// S = Omega * X_act.t() * W * X_act * Omega;

		// double mins = arma::min(s(G)) / 2;
		// S(lower_indices).fill(mins);
		// S(upper_indices).fill(mins);
	    }

	    mat U = arma::chol(S, "upper"); // cholesky decomp, upper tri

	    Ss.push_back(S);
	    Us.push_back(U);
	}

	P = compute_P_chol(X, mu, Us, g, groups);
	while (P.has_nan() || P.has_inf()) {
	    Rcpp::Rcout << "Initialization failed, trying with different initalization.\n";
	    g -= 0.05;
	    if (any(g <= 0)) Rcpp::stop("Failed to initalize\n");
	    P = compute_P_chol(X, mu, Us, g, groups);
	}
    }

    uword num_iter = niter;
    std::vector<double> elbo_values;
    bool converged = false;

    for (unsigned int iter = 1; iter <= niter; ++iter)
    {
	mu_old = mu; g_old = g;

	if (diag_cov) 
	    s_old = s; 

	for (uword i = 0; i < ugroups.size(); ++i)
	{
	    uvec G  = arma::find(groups == ugroups(i));

	    if (diag_cov) 
	    {
		P /= compute_P_G(X, XX, mu, s, g, G);

		mu(G) = pois_update_mu(yX, X, XX, mu, s, lambda, G, P);
		s(G)  = pois_update_s(     X, XX, mu, s, lambda, G, P);

		double tg = pois_update_g(yX, X, XX, mu, s, lambda, w, G, P);
		for (uword j : G) g(j) = tg;

		P %= compute_P_G(X, XX, mu, s, g, G);
	    } 
	    else 
	    {
		mat &U = Us.at(i);
		mat &S = Ss.at(i);
		s_old(G) = s(G);

		P /= compute_P_G_chol(X.cols(G), mu(G), U, g(G(0)));

		mu(G) = pois_update_mu_S(yX(G), X.cols(G), mu(G), U, lambda, P);
		U(trimatu_ind(size(U))) = pois_update_U(X.cols(G), mu(G), U, lambda, P);
		S = U.t() * U;
		
		double tg = pois_update_g_S(yX(G), X.cols(G), mu(G), U, S, lambda, w, P);
		for (uword j : G) g(j) = tg;

		P %= compute_P_G_chol(X.cols(G), mu(G), U, tg);
		s(G) = diagvec(U);
	    }
	}

	if (track_elbo && (iter % track_elbo_every == 0)) {
	    double e = diag_cov ? 
		elbo_poisson(y, X, groups, mu, s, g, P, lambda, w, track_elbo_mcn) :
		elbo_poisson_S(y, X, groups, mu, Us, g, P, lambda, w, track_elbo_mcn);

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
	double e = diag_cov ? 
	    elbo_poisson(y, X, groups, mu, s, g, P, lambda, w, track_elbo_mcn) :
	    elbo_poisson_S(y, X, groups, mu, Us, g, P, lambda, w, track_elbo_mcn);
	elbo_values.push_back(e);
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("S") = Ss,
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


// ----------------------------------------
// Updates under full covariance
// ----------------------------------------

/* Objective function used to test the update eq. for mu */

// double pois_update_mu_obj(const vec &mG, const vec &yX_G, const mat &X_G, const mat &U,
// 	const double lambda, const vec &P)
// {
//     double du = accu(U % U); // trace(S) = sum_ij U_ij^2 
//     const vec PP = P % mvnMGF_chol(X_G, mG, U);

//     double res = - dot(yX_G, mG) +
// 	accu(PP) +
// 	lambda * sqrt(accu(du + mG % mG)); 

//     vec dPPmG = vec(mG.size(), arma::fill::zeros);

//     for (uword j = 0; j < mG.size(); ++j) dPPmG(j) = accu(X_G.col(j) % PP);
    
//     Rcpp::Rcout << dPPmG - yX_G + lambda * mG * pow(du + dot(mG, mG), -0.5);

//     return res;
// }


// -------- update mu ----------
class pois_update_mu_fn_S
{
    public:
	pois_update_mu_fn_S(const vec &yX_G, const mat &X_G, const mat &U, 
		const double lambda, const vec &P) :
	    yX_G(yX_G), X_G(X_G), U(U), lambda(lambda), P(P)
	{
	    du = accu(U % U); 
	};

	double EvaluateWithGradient(const mat &mG, mat &grad)
	{
	    const vec PP = P % mvnMGF_chol(X_G, mG, U);

	    double res = - dot(yX_G, mG) +
		accu(PP) +
		lambda * sqrt(accu(du + mG % mG)); 

	    vec dPPmG = vec(mG.size(), arma::fill::zeros);

	    for (uword j = 0; j < mG.size(); ++j) {
		dPPmG(j) = accu(X_G.col(j) % PP);
	    }
	    
	    grad = dPPmG -
		yX_G +
		lambda * mG * pow(du + dot(mG, mG), -0.5);
	    
	    return res;
	};

    private:
	const vec &yX_G;
	const mat &X_G;
	const mat &U;
	double du;
	const double lambda;
	const vec &P;
};


// [[Rcpp::export]]
vec pois_update_mu_S(const vec &yX_G, const mat &X_G, const vec &mu_G,
	const mat &U, const double lambda, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = GSVB_POS_MAXITS;
    // opt.MaxIterations() = 1000;
    pois_update_mu_fn_S fn(yX_G, X_G, U, lambda, P);

    arma::vec mG = mu_G;
    opt.Optimize(fn, mG);

    return mG;
}


// --------- update S ----------

// double pois_update_U_obj(const mat &U_G, const mat &X_G, const vec &mu_G, const double lambda,
// 	const vec &P)
// {
//     uvec indx = arma::trimatu_ind(size(U_G));
//     mat S = U_G.t() * U_G;
//     mat Sinv = S.i();

//     const double ds = trace(S);

//     const vec PP = P % mvnMGF_chol(X_G, mu_G, U_G);

//     double res = accu(PP) -
// 	0.5 * log(det(S)) +
// 	lambda * pow(ds + dot(mu_G, mu_G), 0.5);
    
//     mat Pgrad = mat(size(U_G), arma::fill::zeros);
//     for (uword i = 0; i < P.size(); ++i) 
//     {
// 	arma::rowvec x = X_G.row(i);
// 	Pgrad += PP(i) * (x.t() * x);
//     }
//     Pgrad = U_G * Pgrad;

//     Pgrad -= U_G * Sinv;
//     Pgrad += lambda * pow(ds + dot(mu_G, mu_G), -0.5) * U_G;
//     Rcpp::Rcout << Pgrad(indx);

//     return res;
// }


class pois_update_U_fn
{
    public:
	pois_update_U_fn(const mat &X_G, const vec &mu_G, const double lambda, 
		const vec &P) :
	    X_G(X_G), mu_G(mu_G), lambda(lambda), P(P)
	{
	    mk = X_G.n_cols;
	    U = mat(mk, mk, arma::fill::zeros);
	    indx = arma::trimatu_ind(size(U));
	};

	double EvaluateWithGradient(const mat &u, mat &grad)
	{
	    U(indx) = u;
	    mat S = U.t() * U;
	    mat Sinv = S.i();

	    const double ds = trace(S);

	    const vec PP = P % mvnMGF_chol(X_G, mu_G, U);

	    double res = accu(PP) -
		0.5 * log(det(S)) +
		lambda * pow(ds + dot(mu_G, mu_G), 0.5);

	    mat Pgrad = mat(size(U), arma::fill::zeros);
	    for (uword i = 0; i < P.size(); ++i) 
	    {
		arma::rowvec x = X_G.row(i);
		Pgrad += PP(i) * (x.t() * x);
	    }
	    Pgrad = U * Pgrad;

	    Pgrad -= U * Sinv;
	    Pgrad += lambda * pow(ds + dot(mu_G, mu_G), -0.5) * U;

	    grad = Pgrad(indx);
	    
	    return res;
	};


    private:
	const mat &X_G;
	const vec &mu_G;
	const double lambda;
	const vec &P;
	double mk;
	mat U;
	uvec indx;
};


// [[Rcpp::export]]
vec pois_update_U(const mat &X_G, const vec &mu_G, const mat &U,
	const double lambda, const vec &P)
{
    ens::L_BFGS opt;
    opt.MaxIterations() = GSVB_POS_MAXITS;
    // opt.MaxIterations() = 1000;
    pois_update_U_fn fn(X_G, mu_G, lambda, P);

    arma::vec ug = U(trimatu_ind(size(U)));
    opt.Optimize(fn, ug);

    return ug;
}


// --------- update g ----------
// [[Rcpp::export]]
double pois_update_g_S(const vec &yX_G, const mat &X_G, const vec &mu_G,
	const mat &U, const mat &S, const double lambda, const double w,
	const vec &P)
{
    const double mk = X_G.n_cols;
    const double Ck = mk * log(2.0) + 0.5*(mk-1.0)*log(M_PI) + 
	lgamma(0.5*(mk + 1.0));

    double ds = trace(S);

    const vec P1 = mvnMGF_chol(X_G, mu_G, U);

    const double res =
	log(w / (1 - w)) + 
	0.5 * mk - 
	Ck +
	mk * log(lambda) +
	0.5 * log(det(2.0 * M_PI * S)) -
	lambda * sqrt(ds + dot(mu_G, mu_G)) +
	dot(yX_G, mu_G) -
	sum(P % (P1 - 1));

    return 1.0/(1.0 + exp(-res));
}


// [[Rcpp::export]]
double elbo_poisson_S(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const std::vector<mat> &Ss, const vec &g, 
	const double lambda, const double w, const uword mcn)
{
    std::vector<mat> Us;
    for (mat S : Ss) {
	Us.push_back(chol(S, "upper"));
    }
    const vec P = compute_P_chol(X, mu, Us, g, groups);
    double res = elbo_poisson_S(y, X, groups, mu, Us, g, P, lambda, w, mcn);

    return(res);
}


double elbo_poisson_S(const vec &y, const mat &X, const uvec &groups,
	const vec &mu, const std::vector<mat> &Us, const vec &g, 
	const vec &P, const double lambda, const double w, const uword mcn)
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
	for (uword gi = 0; gi < ugroups.size(); ++gi)
	{
	    uvec G = find(groups == ugroups(gi));
	    uword k = G(0);
	    double mk = G.size();

	    // Compute the Monte-Carlo integral of E_Q [ lambda * || b_{G_k} || ]
	    vec beta_G = Us.at(gi) * arma::randn(mk) + mu(G);
	    mci -= lambda * g(k) * norm(beta_G, 2);
	}
    }
    mci = mci / static_cast<double>(mcn);
    res += mci;

    return(res);
}


