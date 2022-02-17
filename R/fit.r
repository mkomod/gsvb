#' Fit high-dimensional group-sparse regression models
#'
#' @param y response vector
#' @param X input matrix
#' @param groups group structure
#' @param lambda penalization hyperparameter for the multivariate exponential prior
#' @param a0 shape parameter for the Beta(a0, b0) mixing prior
#' @param b0 shape parameter for the Beta(a0, b0) mixing prior
#' @param sigma TODO 
#' @param mu initial values of mu, the means of the variational family
#' @param s initial values of s, the std. dev of the variational family
#' @param g initial values of g, the group inclusion probabilities of the variational family
#' @param track_elbo track the evidence lower bound (ELBO)
#' @param track_elbo_every the number of iterations between computing the ELBO
#' @param track_elbo_mcn number of Monte-Carlo samples to compute the ELBO
#' @param niter maximum number of iteration to run the algorithm for
#' @param tol convergence tolerance
#' @param verbose print additional information
#' 
#' @return The program output is a list containing:
#' \item{mu}{the means for the variational posterior}
#' \item{s}{the std. dev. for the variational posterior}
#' \item{g}{the group inclusion probabilities}
#' \item{beta_hat}{the variational posterior mean}
#' \item{parameters}{a list containing the model hyperparameters}
#' \item{converged}{a boolean indicating if the algorithm has converged}
#' \item{iter}{the number of iterations the algorithm was ran for}
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
#' 
#' plot(f$beta_hat, col=4, ylab=expression(hat(beta)))
#' points(b, pch=20)
#'
#'
#' @export
gsvb.fit <- function(y, X, groups, lambda=1, a0=1, 
    b0=length(unique(groups)), sigma=1, mu=runif(ncol(X), -0.2, 0.2), 
    s=apply(X, 2, function(x) 1/sqrt(sum(x^2) / sigma^2 + 2*lambda)),
    g=rep(0.5, ncol(X)), track_elbo=TRUE, track_elbo_every=5, 
    track_elbo_mcn=1e4, niter=150, tol=1e-3, verbose=TRUE) 
{
    # pre-processing
    group.order <- order(groups)
    groups <- groups[group.order]
    X <- X[ , group.order]

    # run algorithm
    f <- fit(y, X, groups, lambda, a0, b0, sigma, mu, s, g, track_elbo, 
	track_elbo_every, track_elbo_mcn, niter, tol, verbose)

    # re-order to match input order
    mu <- f$mu[group.order]
    s <-  f$s[group.order]
    g <-  f$g[group.order]
   
    res <- list(
	mu = mu,
	s = s,
	g = g,
	beta_hat = mu * g,
	parameters = list(lambda = lambda, a0 = a0, b0=b0, sigma=sigma),
	converged = f$converged,
	iter = f$iter
    )
   
    if (track_elbo)
	res$elbo <- f$elbo

    return(res)
}

