#include "rmath.h"

#include <cmath>

namespace gsvb {
namespace rmath {

namespace {

const double SQRT1_2     = 0.7071067811865475244;   // 1 / sqrt(2)
const double SQRT_PI     = 1.7724538509055160273;   // sqrt(pi)
const double LOG_HALF    = -0.6931471805599453094;  // log(0.5)
const double HALF_LOG2PI = 0.9189385332046727418;   // 0.5 * log(2*pi)

// Scaled complementary error function erfcx(x) = exp(x^2) * erfc(x), for x >= 0.
// For moderate x the direct product is accurate and well within range
// (exp(13^2) ~ 1e73). For larger x we use the asymptotic series, which is
// accurate to ~1e-11 (relative) from x = 13 upward and improves thereafter.
double erfcx_pos(double x)
{
    if (x <= 13.0) {
        return std::exp(x * x) * std::erfc(x);
    }
    const double r = 1.0 / (x * x);
    // 1 - r/2 + 3r^2/4 - 15r^3/8 + 105r^4/16
    const double s = 1.0 - r * (0.5 - r * (0.75 - r * (1.875 - r * 6.5625)));
    return s / (x * SQRT_PI);
}

} // anonymous namespace


double lgammafn(double x)
{
    return std::lgamma(x);
}


double digamma(double x)
{
    double result = 0.0;

    // recurrence psi(x) = psi(x+1) - 1/x  until x is large enough for the
    // asymptotic expansion to be accurate to ~1e-13
    while (x < 12.0) {
        result -= 1.0 / x;
        x += 1.0;
    }

    const double r  = 1.0 / x;
    const double r2 = r * r;

    // psi(x) ~ log(x) - 1/(2x) - 1/(12 x^2) + 1/(120 x^4) - 1/(252 x^6) + 1/(240 x^8)
    result += std::log(x) - 0.5 * r;
    result -= r2 * (1.0 / 12.0 -
              r2 * (1.0 / 120.0 -
              r2 * (1.0 / 252.0 -
              r2 * (1.0 / 240.0))));

    return result;
}


double trigamma(double x)
{
    double result = 0.0;

    // recurrence psi'(x) = psi'(x+1) + 1/x^2
    while (x < 12.0) {
        result += 1.0 / (x * x);
        x += 1.0;
    }

    const double r  = 1.0 / x;
    const double r2 = r * r;

    // psi'(x) ~ 1/x + 1/(2 x^2) + 1/(6 x^3) - 1/(30 x^5) + 1/(42 x^7)
    result += r + 0.5 * r2 +
              r * r2 * (1.0 / 6.0 -
              r2 * (1.0 / 30.0 -
              r2 * (1.0 / 42.0)));

    return result;
}


double dnorm(double x, double mu, double sigma, int give_log)
{
    const double z  = (x - mu) / sigma;
    const double ld = -0.5 * z * z - std::log(sigma) - HALF_LOG2PI;
    return give_log ? ld : std::exp(ld);
}


double pnorm(double x, double mu, double sigma, int lower_tail, int log_p)
{
    const double z = (x - mu) / sigma;

    // Requested tail probability equals 0.5 * erfc(q).
    //   lower tail  Phi(z)     = 0.5 * erfc(-z / sqrt(2))
    //   upper tail  1 - Phi(z) = 0.5 * erfc( z / sqrt(2))
    const double q = (lower_tail ? -z : z) * SQRT1_2;

    if (!log_p) {
        return 0.5 * std::erfc(q);
    }

    // log of the tail, computed stably so the far tail does not underflow to -inf
    if (q <= 0.0) {
        // erfc(q) is in [1, 2); direct log is safe
        return LOG_HALF + std::log(std::erfc(q));
    }
    // erfc(q) = erfcx(q) * exp(-q^2)  =>  log = log(erfcx(q)) - q^2
    return LOG_HALF + std::log(erfcx_pos(q)) - q * q;
}

} // namespace rmath
} // namespace gsvb
