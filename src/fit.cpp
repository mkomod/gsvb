#include "fit.h"


// [[Rcpp::export]]
Rcpp::List fit(vec y, mat X, veci groups, vec mu, vec s, vec g,
	double a0, double b0, double lambda, unsigned int niter, 
	double tol, bool verbose, double sigma) 
{
    const int n = X.rows();
    const int p = X.cols();

    mat xtx = X.transpose() * X;
    vec yx = y.transpose() * X;

    double w = a0 / (a0 + b0);
   
    for (int i = 0; i < p; ++i)
	s[i] = pow(xtx(i, i) / pow(sigma, 2.0) + 2 * lambda, -0.5);

    vec mu_old, s_old, g_old;
   
    // unique groups
    std::set<int> ugroups(groups.data(), groups.data() + groups.size());
    auto gindices = get_group_indices(groups);
    
    for (int iter = 1; iter <= niter; ++iter) {

	mu_old = mu; s_old = s; g_old = g;
	
	// group wise updates
	for (int gi = 0; gi < ugroups.size(); ++gi) {
	    int gi_beg = gindices.at(gi).at(0);
	    int gi_end = gindices.at(gi).at(1);

	    // update mu
	    for (int i = gi_beg; i <= gi_end; ++i)
		mu(i) = update_mu(i, gi, xtx, yx(i), mu, s, g, sigma, lambda, gindices);

	    // update gamma
	    double tg = update_g(gi, xtx, yx, mu, s, g, sigma, lambda, w, gindices);
	    for (int i = gi_beg; i <= gi_end; ++i) g(i) = tg;
	}
	
	// check from break
	Rcpp::checkUserInterrupt();
	Rcpp::Rcout << iter;

	// check convergence
	if ((mu - mu_old).cwiseAbs().sum() < tol && 
	    (s - s_old).cwiseAbs().sum() < tol && 
	    (g - g_old).cwiseAbs().sum() < tol) {
	    if (verbose)
		Rcpp::Rcout << "Converged in " << iter << " iterations\n";
	    return Rcpp::List::create(
		Rcpp::Named("mu") = mu,
		Rcpp::Named("sigma") = s,
		Rcpp::Named("gamma") = g,
		Rcpp::Named("converged") = true,
		Rcpp::Named("iter") = iter
	    );
	}
    }

    return Rcpp::List::create(
	Rcpp::Named("mu") = mu,
	Rcpp::Named("sigma") = s,
	Rcpp::Named("gamma") = g,
	Rcpp::Named("converged") = false,
	Rcpp::Named("iter") = niter
    );
}


// solveable optimization for lower upper bound
inline double update_mu(unsigned int i, int gi, const mat &xtx, double yx_i, 
	const vec &mu, const vec &s, const vec &g, double sigma, double lambda,
	const std::vector<std::array<int, 2>> &gindices)
{
    double res = yx_i;
    for (int grp = 0; grp < gindices.size(); grp++) {
	if (grp == gi) {
	    for (int j = gindices.at(gi).at(0); j <= gindices.at(gi).at(1); ++j) {
		if (j == i) continue;
		res -= 0.5 * xtx(j, i) * mu(j);
	    }
	} else {
	    for (int j = gindices.at(grp).at(0); j <= gindices.at(grp).at(1); ++j)
		res -= xtx(j, i) * g(j) * mu(j);
	}
    }
    res /= (xtx(i, i) + 2 * pow(sigma, 2.0) * lambda);
    // Rcpp::Rcout << "mu" << i << ": " << res << "\n";

    return res;
}



inline double update_g(unsigned int gi, const mat &xtx, const vec &yx, const vec &mu,
	const vec &s, const vec &g, double sigma, double lambda, double w,
	const std::vector<std::array<int, 2>> &gindices) 
{
    double a = 0;
    double c = 0.5 * pow(sigma, -2.0);
    unsigned int mk = gindices.at(gi).at(1) - gindices.at(gi).at(0) + 1;

    double res = log(w / (1.0 - w)) + mk / 2.0;

    // normalizing constants
    res -= (mk * log(2) + 0.5 * (mk - 1.0) * log(M_PI) + R::lgammafn(0.5 * (mk + 1)));
    res += mk * log(lambda);
    res += 0.5 * mk * log(2.0 * M_PI);

    for (int i = gindices.at(gi).at(0); i <= gindices.at(gi).at(1); ++i) {
	// normalizing constant variational family
	res += log(s(i));

	// 1/sigma^2 <y, X mu>
	res += pow(sigma, -2.0) * yx(i) * mu(i);

	// lambda * sum(s^2 + mu^2)
	a += (pow(s(i), 2.0) + pow(mu(i), 2.0));

	res -= c * xtx(i, i) * pow(s(i), 2.0);

	for (int j = gindices.at(gi).at(0); j <= gindices.at(gi).at(1); ++j)
	    res -= c * xtx(j, i) * mu(j) * mu(i);

	for (int ngi = 0; ngi < gindices.size(); ++ngi) {
	    if (ngi == gi) continue;
	    for (int j = gindices.at(ngi).at(0); j <= gindices.at(ngi).at(1); ++j)
		res -= 2.0 * c * xtx(j, i) * g(j) * mu(j) * mu(i);
	}
    }
    res -= lambda * sqrt(a);

    return sigmoid(res);
}
