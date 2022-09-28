#' Compute the Evidence Lower Bound (ELBO)
#'
#' @param fit the fit model.
#' @param y response vector.
#' @param X input matrix.
#' @param mcn number of Monte-Carlo samples.
#' @param approx elements of gamma less than an approximation threshold are not used in computations.
#' @param approx_thresh the threshold below which elements of gamma are not used.
#'
#' @return the ELBO (numeric)
#' 
#' @section Details: TODO
#'
#' @examples
#' n <- 100
#' p <- 1000
#' gsize <- 5
#' groups <- c(rep(1:(p/gsize), each=gsize))
#' 
#' X <- matrix(rnorm(n * p), nrow=n, ncol=p)
#' b <- c(rep(0, gsize), rep(-4, gsize), rep(8, gsize), rep(0, p - 3 * gsize))
#' y <- X %*% b + rnorm(n, 0, 1)
#' 
#' f <- gsvb.fit(y, X, groups)
#' gsvb.elbo(f, y, X, groups) 
#'
#' @export
gsvb.elbo <- function(fit, y, X, mcn=5e2, approx=FALSE, approx_thresh=1e-3)
{
    if (fit$parameters$intercept) {
	X <- cbind(rep(1, nrow(X)), X)
    }
    groups <- fit$parameters$groups

    yty <- sum(y * y)
    yx <- t(X) %*% y
    xtx <- t(X) %*% X

    n <- nrow(X)
    p <- ncol(X)

    res <- ifelse(fit$parameters$diag_covariance,

	elbo_linear_c(yty, yx, xtx, groups, n, p, fit$mu, fit$s, fit$g[groups],
	fit$tau_a, fit$tau_b, fit$parameters$lambda, fit$parameters$a0, 
	fit$parameters$b0, fit$parameters$tau_a0, fit$parameters$tau_b0,
	mcn, approx, approx_thresh),

	elbo_linear_u(yty, yx, xtx, groups, n, p, fit$mu, fit$s, fit$g[groups],
	fit$tau_a, fit$tau_b, fit$parameters$lambda, fit$parameters$a0, 
	fit$parameters$b0, fit$parameters$tau_a0, fit$parameters$tau_b0,
	mcn, approx, approx_thresh)
    )

    return(res)
}
