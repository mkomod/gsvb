#ifndef GSVB_RMATH_H
#define GSVB_RMATH_H

// Self-contained replacements for the handful of R math-library functions
// used by the GSVB backend. These mirror the semantics of the corresponding
// R `R::` entry points (argument order and the log-tail conventions of
// pnorm/dnorm) so the algorithm code can call them unchanged via the shim in
// gsvb_compat.h. Implemented from <cmath> only -- no external dependencies.

namespace gsvb {
namespace rmath {

// log Gamma(x)
double lgammafn(double x);

// digamma  psi(x)   = d/dx log Gamma(x)
double digamma(double x);

// trigamma psi'(x)  = d^2/dx^2 log Gamma(x)
double trigamma(double x);

// Normal density. dnorm(x, mu, sigma, log_p).
// Matches R::dnorm / R::dnorm4 (dnorm4 is the C entry point for dnorm).
double dnorm(double x, double mu, double sigma, int give_log);

// Normal CDF. pnorm(x, mu, sigma, lower_tail, log_p).
// When log_p != 0 a numerically stable log of the (possibly tiny) tail is
// returned -- this path is used heavily by logistic.cpp and must not
// underflow to -inf.
double pnorm(double x, double mu, double sigma, int lower_tail, int log_p);

} // namespace rmath
} // namespace gsvb

#endif
