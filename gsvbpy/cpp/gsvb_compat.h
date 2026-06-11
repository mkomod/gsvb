#ifndef GSVB_COMPAT_H
#define GSVB_COMPAT_H

// Compatibility shim that lets the original GSVB algorithm code (linear.cpp,
// logistic.cpp, poisson.cpp, utils.cpp) compile unchanged outside of R/Rcpp.
// It supplies:
//   (a) an `R::` namespace forwarding to the self-contained rmath functions,
//   (b) `log1pexp` as an in-place functor for arma::Col::for_each,
//   (c) a minimal `Rcpp` shim (Rcout / stop / checkUserInterrupt).
//
// Must be included after <armadillo> (it uses arma::randu).

#include <armadillo>
#include <iostream>
#include <stdexcept>
#include <string>

#include "rmath.h"

// (a) R math library entry points used by the backend ------------------------
namespace R {

inline double lgammafn(double x) { return gsvb::rmath::lgammafn(x); }
inline double digamma(double x)  { return gsvb::rmath::digamma(x); }
inline double trigamma(double x) { return gsvb::rmath::trigamma(x); }

// pnorm(x, mean, sd, lower_tail, log_p)
inline double pnorm(double x, double mu, double sigma, int lower_tail, int log_p)
{
    return gsvb::rmath::pnorm(x, mu, sigma, lower_tail, log_p);
}

// dnorm(x, mean, sd, log)  -- dnorm4 is R's C entry point for the same function
inline double dnorm(double x, double mu, double sigma, int give_log)
{
    return gsvb::rmath::dnorm(x, mu, sigma, give_log);
}
inline double dnorm4(double x, double mu, double sigma, int give_log)
{
    return gsvb::rmath::dnorm(x, mu, sigma, give_log);
}

// runif(a, b) -- uses Armadillo's RNG (seedable via gsvb set_seed)
inline double runif(double a, double b) { return a + (b - a) * arma::randu(); }

} // namespace R


// (b) log1pexp ---------------------------------------------------------------
// Stable log(1 + exp(x)), written as an in-place functor so it can be passed to
// arma::Col<double>::for_each (which requires the signature void(double&)).
inline void log1pexp(double& x)
{
    if (x <= -37.0)      x = std::exp(x);
    else if (x <= 18.0)  x = std::log1p(std::exp(x));
    else if (x <= 33.3)  x = x + std::exp(-x);
    // else: x is already a good approximation of log(1+exp(x))
}


// (c) minimal Rcpp shim ------------------------------------------------------
namespace Rcpp {

// defined once in bindings.cpp as `std::ostream& Rcpp::Rcout = std::cout;`
extern std::ostream& Rcout;

inline void checkUserInterrupt() {}

[[noreturn]] inline void stop(const std::string& msg)
{
    throw std::runtime_error(msg);
}

} // namespace Rcpp

#endif
