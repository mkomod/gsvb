#include "RcppArmadillo.h"
#include "gsvb_types.h"
#include <vector>
#include <set>

void f(std::vector<mat> &vec) {
    for (int i = 0; i < 2; ++i) {
	mat S = vec.at(i);
	Rcpp::Rcout << S;
    }
}

// [[Rcpp::export]]
SEXP mats()
{
    std::vector<mat> s;	
    for (int i = 0; i < 2; ++i) {
	mat b = arma::mat(2, 2, arma::fill::randn);
	s.push_back(b);
    }

    for (int i = 0; i < 2; ++i) {
	Rcpp::Rcout << s.at(i);
	Rcpp::Rcout << "\n";
    }

    // mat &c = s.at(0); // note: we need a refference so we can edit c
    // c(0, 0) = 1.0;
    // Rcpp::Rcout << c << "\n";
    // Rcpp::Rcout << s.at(0) << "\n";
    // Rcpp::Rcout << c.at(0, 0) << "\n";

    f(s);
    
    return Rcpp::wrap(s);
}


void g(mat a) {
   Rcpp::Rcout << a(1, 1);
}

void g(vec a) {
   Rcpp::Rcout << a(0);
}

// [[Rcpp::export]]





