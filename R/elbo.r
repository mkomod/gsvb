#' Compute the Evidence Lower Bound (ELBO)
#'
#' @param fit fit model
#' @param y response vector
#' @param X input matrix
#' @param groups group structure
#' @param mcn number of Monte-Carlo samples
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
gsvb.elbo <- function(fit, y, X, groups, mcn=1e4) 
{
    return(elbo(y, X, groups, fit$mu, fit$s, fit$g, fit$parameters$lambda, 
	fit$parameters$a0, fit$parameters$b0, fit$parameters$sigma, mcn))
}

