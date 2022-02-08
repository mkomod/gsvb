#include "RcppEigen.h"

#include "gsvb_types.h"


// [[Rcpp::export]]
double test(Rcpp::NumericMatrix a) {
    double b = sqrt(Rcpp::sum(Rcpp::pow(a, 2.0)));

    return 0.0;
}

