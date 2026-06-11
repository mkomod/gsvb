// pybind11 module for the GSVB backend.
//
// All numpy <-> Armadillo conversion lives here so the algorithm translation
// units never see pybind11. Conversions copy (no zero-copy / borrowing), which
// keeps memory ownership entirely on either the numpy or the Armadillo side and
// avoids any allocator coupling between translation units.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // std::vector<double> -> list
#include <pybind11/numpy.h>

#include <algorithm>
#include <iostream>
#include <vector>
#include <armadillo>

#include "gsvb_results.h"
#include "rmath.h"
#include "linear.h"
#include "logistic.h"
#include "poisson.h"

namespace py = pybind11;

// single definition of the Rcout symbol declared in gsvb_compat.h
std::ostream& Rcpp::Rcout = std::cout;

namespace {

using arr_d = py::array_t<double, py::array::c_style | py::array::forcecast>;
using arr_u = py::array_t<uint64_t, py::array::c_style | py::array::forcecast>;

// ---- numpy -> Armadillo (copying) ----
arma::vec to_col(const arr_d& a)
{
    auto info = a.request();
    const double* p = static_cast<const double*>(info.ptr);
    const arma::uword n = static_cast<arma::uword>(info.size);
    arma::vec v(n);
    std::copy(p, p + n, v.memptr());
    return v;
}

arma::uvec to_uvec(const arr_u& a)
{
    auto info = a.request();
    const uint64_t* p = static_cast<const uint64_t*>(info.ptr);
    const arma::uword n = static_cast<arma::uword>(info.size);
    arma::uvec v(n);
    for (arma::uword i = 0; i < n; ++i) v[i] = static_cast<arma::uword>(p[i]);
    return v;
}

arma::mat to_mat(const arr_d& a)
{
    auto info = a.request();
    if (info.ndim != 2) throw std::runtime_error("gsvbpy: expected a 2-D array");
    const arma::uword nr = static_cast<arma::uword>(info.shape[0]);
    const arma::uword nc = static_cast<arma::uword>(info.shape[1]);
    const double* p = static_cast<const double*>(info.ptr);   // C-order: p[i*nc + j]
    arma::mat M(nr, nc);
    for (arma::uword i = 0; i < nr; ++i)
        for (arma::uword j = 0; j < nc; ++j)
            M(i, j) = p[i * nc + j];
    return M;
}

std::vector<arma::mat> to_vec_mat(const py::list& lst)
{
    std::vector<arma::mat> out;
    out.reserve(lst.size());
    for (auto item : lst) out.push_back(to_mat(item.cast<arr_d>()));
    return out;
}

// ---- Armadillo -> numpy (copying) ----
py::array_t<double> from_col(const arma::vec& v)
{
    py::array_t<double> out(static_cast<py::ssize_t>(v.n_elem));
    std::copy(v.memptr(), v.memptr() + v.n_elem,
              static_cast<double*>(out.request().ptr));
    return out;
}

py::array_t<double> from_mat(const arma::mat& M)
{
    py::array_t<double> out({static_cast<py::ssize_t>(M.n_rows),
                             static_cast<py::ssize_t>(M.n_cols)});
    auto r = out.mutable_unchecked<2>();
    for (arma::uword i = 0; i < M.n_rows; ++i)
        for (arma::uword j = 0; j < M.n_cols; ++j)
            r(i, j) = M(i, j);
    return out;
}

py::list from_vec_mat(const std::vector<arma::mat>& Ss)
{
    py::list out;
    for (const auto& S : Ss) out.append(from_mat(S));
    return out;
}

// ---- result structs -> dict ----
py::dict to_dict(const LinearFit& r)
{
    py::dict d;
    d["mu"]         = from_col(r.mu);
    d["sigma"]      = from_col(r.sigma);
    d["S"]          = from_vec_mat(r.S);
    d["gamma"]      = from_col(r.gamma);
    d["tau_a"]      = r.tau_a;
    d["tau_b"]      = r.tau_b;
    d["converged"]  = r.converged;
    d["iterations"] = r.iterations;
    d["elbo"]       = r.elbo;
    return d;
}

py::dict to_dict(const GlmFit& r)
{
    py::dict d;
    d["mu"]         = from_col(r.mu);
    d["sigma"]      = from_col(r.sigma);
    d["gamma"]      = from_col(r.gamma);
    d["S"]          = from_vec_mat(r.S);
    d["converged"]  = r.converged;
    d["iterations"] = r.iterations;
    d["elbo"]       = r.elbo;
    return d;
}

} // namespace


PYBIND11_MODULE(_gsvb_core, m)
{
    m.doc() = "GSVB C++ backend (Armadillo + Ensmallen) bound via pybind11";

    typedef arma::uword uword;

    m.def("set_seed", [](uint64_t seed) { arma::arma_rng::set_seed(seed); },
          py::arg("seed"));

    // exposed for unit testing the self-contained special functions
    m.def("_digamma",  &gsvb::rmath::digamma);
    m.def("_trigamma", &gsvb::rmath::trigamma);
    m.def("_pnorm",    &gsvb::rmath::pnorm);
    m.def("_dnorm",    &gsvb::rmath::dnorm);

    // ------------------------------ linear ------------------------------
    m.def("fit_linear",
        [](arr_d y, arr_d X, arr_u groups, double lambda, double a0, double b0,
           double tau_a0, double tau_b0, arr_d mu, arr_d s, arr_d g,
           bool diag_cov, bool track_elbo, uword track_elbo_every,
           uword track_elbo_mcn, unsigned int niter, double tol, bool verbose,
           uword ordering) {
            return to_dict(fit_linear(to_col(y), to_mat(X), to_uvec(groups),
                lambda, a0, b0, tau_a0, tau_b0, to_col(mu), to_col(s), to_col(g),
                diag_cov, track_elbo, track_elbo_every, track_elbo_mcn, niter,
                tol, verbose, ordering));
        });

    m.def("elbo_linear_c",
        [](double yty, arr_d yx, arr_d xtx, arr_u groups, uword n, uword p,
           arr_d mu, arr_d s, arr_d g, double tau_a, double tau_b, double lambda,
           double a0, double b0, double tau_a0, double tau_b0, uword mcn,
           bool approx, double approx_thresh) {
            return elbo_linear_c(yty, to_col(yx), to_mat(xtx), to_uvec(groups),
                n, p, to_col(mu), to_col(s), to_col(g), tau_a, tau_b, lambda, a0,
                b0, tau_a0, tau_b0, mcn, approx, approx_thresh);
        });

    m.def("elbo_linear_u",
        [](double yty, arr_d yx, arr_d xtx, arr_u groups, uword n, uword p,
           arr_d mu, py::list Ss, arr_d g, double tau_a, double tau_b,
           double lambda, double a0, double b0, double tau_a0, double tau_b0,
           uword mcn, bool approx, double approx_thresh) {
            return elbo_linear_u(yty, to_col(yx), to_mat(xtx), to_uvec(groups),
                n, p, to_col(mu), to_vec_mat(Ss), to_col(g), tau_a, tau_b,
                lambda, a0, b0, tau_a0, tau_b0, mcn, approx, approx_thresh);
        });

    // ------------------------------ logistic ----------------------------
    m.def("fit_logistic",
        [](arr_d y, arr_d X, arr_u groups, double lambda, double a0, double b0,
           arr_d mu, arr_d s, arr_d g, bool diag_cov, bool track_elbo,
           uword track_elbo_every, uword track_elbo_mcn, double thresh, int l,
           unsigned int niter, unsigned int alg, double tol, bool verbose,
           uword ordering) {
            return to_dict(fit_logistic(to_col(y), to_mat(X), to_uvec(groups),
                lambda, a0, b0, to_col(mu), to_col(s), to_col(g), diag_cov,
                track_elbo, track_elbo_every, track_elbo_mcn, thresh, l, niter,
                alg, tol, verbose, ordering));
        });

    m.def("elbo_logistic",
        [](arr_d y, arr_d X, arr_u groups, arr_d mu, arr_d s, arr_d g,
           py::list Ss, double lambda, double w, uword mcn, bool diag) {
            return elbo_logistic(to_col(y), to_mat(X), to_uvec(groups),
                to_col(mu), to_col(s), to_col(g), to_vec_mat(Ss), lambda, w,
                mcn, diag);
        });

    // ------------------------------ poisson -----------------------------
    m.def("fit_poisson",
        [](arr_d y, arr_d X, arr_u groups, double lambda, double a0, double b0,
           arr_d mu, arr_d s, arr_d g, bool diag_cov, bool track_elbo,
           uword track_elbo_every, uword track_elbo_mcn, unsigned int niter,
           double tol, bool verbose) {
            return to_dict(fit_poisson(to_col(y), to_mat(X), to_uvec(groups),
                lambda, a0, b0, to_col(mu), to_col(s), to_col(g), diag_cov,
                track_elbo, track_elbo_every, track_elbo_mcn, niter, tol,
                verbose));
        });

    m.def("elbo_poisson",
        [](arr_d y, arr_d X, arr_u groups, arr_d mu, arr_d s, arr_d g,
           double lambda, double w, uword mcn) {
            return elbo_poisson(to_col(y), to_mat(X), to_uvec(groups),
                to_col(mu), to_col(s), to_col(g), lambda, w, mcn);
        });

    m.def("elbo_poisson_S",
        [](arr_d y, arr_d X, arr_u groups, arr_d mu, py::list Ss, arr_d g,
           double lambda, double w, uword mcn) {
            return elbo_poisson_S(to_col(y), to_mat(X), to_uvec(groups),
                to_col(mu), to_vec_mat(Ss), to_col(g), lambda, w, mcn);
        });

    m.def("pois_update_mu_S",
        [](arr_d yX_G, arr_d X_G, arr_d mu_G, arr_d U, double lambda, arr_d P) {
            return from_col(pois_update_mu_S(to_col(yX_G), to_mat(X_G),
                to_col(mu_G), to_mat(U), lambda, to_col(P)));
        });

    m.def("pois_update_U",
        [](arr_d X_G, arr_d mu_G, arr_d U, double lambda, arr_d P) {
            return from_col(pois_update_U(to_mat(X_G), to_col(mu_G), to_mat(U),
                lambda, to_col(P)));
        });

    m.def("pois_update_g_S",
        [](arr_d yX_G, arr_d X_G, arr_d mu_G, arr_d U, arr_d S, double lambda,
           double w, arr_d P) {
            return pois_update_g_S(to_col(yX_G), to_mat(X_G), to_col(mu_G),
                to_mat(U), to_mat(S), lambda, w, to_col(P));
        });

    // ------------------------------ utils -------------------------------
    m.def("mvnMGF",
        [](arr_d X, arr_d mu, arr_d S) {
            return from_col(mvnMGF(to_mat(X), to_col(mu), to_mat(S)));
        });

    m.def("mvnMGF_chol",
        [](arr_d X, arr_d mu, arr_d U) {
            return from_col(mvnMGF_chol(to_mat(X), to_col(mu), to_mat(U)));
        });
}
