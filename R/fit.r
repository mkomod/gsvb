#' Fit high-dimensional group-sparse regression models
#'
#' @param y response vector.
#' @param X input matrix.
#' @param groups group structure.
#' @param intercept should an intercept term be included.
#' @param diag_covariance should a diagonal covariance matrix be used in the variational approximation. Note: if true then the *standard deviations* for each coefficient are returned. If false then covariance matrices for each group are returned.
#' @param lambda penalization hyperparameter for the multivariate exponential prior.
#' @param a0 shape parameter for the Beta(a0, b0) mixing prior.
#' @param b0 shape parameter for the Beta(a0, b0) mixing prior.
#' @param tau_a0 shape parameter for the inverse-Gamma(a0, b0) prior on the variance, tau^2.
#' @param tau_b0 scale parameter for the inverse-Gamma(a0, b0) prior on the variance, tau^2.
#' @param mu initial values of mu, the means of the variational family.
#' @param s initial values of s, the std. dev of the variational family.
#' @param g initial values of g, the group inclusion probabilities of the variational family.
#' @param track_elbo track the evidence lower bound (ELBO).
#' @param track_elbo_every the number of iterations between computing the ELBO.
#' @param track_elbo_mcn number of Monte-Carlo samples to compute the ELBO.
#' @param niter maximum number of iteration to run the algorithm for.
#' @param tol convergence tolerance.
#' @param verbose print additional information.
#' 
#' @return The program output is a list containing:
#' \item{mu}{the means for the variational posterior.}
#' \item{s}{the std. dev. or covaraince matrices for the variational posterior.}
#' \item{g}{the group inclusion probabilities.}
#' \item{beta_hat}{the variational posterior mean.}
#' \item{tau_hat}{the mean of the variance term.}
#' \item{tau_a}{the shape parameter for the variational posterior of tau^2. This is an inverse-Gamma(tau_a0, tau_b0) distribtuion.}
#' \item{tau_b}{the scale parameter for the variational posterior of tau^2.}
#' \item{parameters}{a list containing the model hyperparameters.}
#' \item{converged}{a boolean indicating if the algorithm has converged.}
#' \item{iter}{the number of iterations the algorithm was ran for.}
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
gsvb.fit <- function(y, X, groups, intercept=TRUE, diag_covariance=TRUE,
    lambda=1, a0=1, b0=length(unique(groups)), tau_a0=1e-3, tau_b0=1e-3,
    mu=NULL, s=apply(X, 2, function(x) 1/sqrt(sum(x^2)*tau_a0/tau_b0+2*lambda)),
    g=rep(0.5, ncol(X)), track_elbo=TRUE, track_elbo_every=5, 
    track_elbo_mcn=5e2, niter=150, tol=1e-3, verbose=TRUE) 
{
    # check user input
    if (min(groups) != 1) 
	stop("group labels must start at 1")
    if (max(groups) != length(unique(groups)))
	stop("group labels must not exceed the unique number of groups")
    if (!all(groups == rep(unique(groups), table(groups))))
	stop("groups must be ordered")
    if (!is.matrix(X))
	stop("X must be a matrix")
    if (any(c(lambda, a0, b0, tau_a0, tau_b0) <= 0))
	stop("Hyperparameters must be greater than 0")

    # pre-processing
    if (intercept) {
	groups <- c(min(groups) - 1, groups) + 1
	X <- cbind(rep(1, nrow(X)), X)
	
	# update the initial parameters
	if (length(s) == ncol(X) - 1)
	    s <- c(1/sqrt(sqrt(n) * tau_a0 / tau_b0 + 2 *lambda), s)

	if (length(g) != ncol(X))
	    g <- c(0.5, g)
    }
    
    if (is.null(mu)) {
	# Note: the intercept is handled by adding a column of 1s to the
	# design matrix X and is therefore disabled for gglasso
	glfit <- gglasso::gglasso(X, y, groups, nlambda=10, intercept=FALSE)

	# take mu as the estimate for smallest reg parameter
	mu <- glfit$beta[ , length(glfit$lambda)]
    }

    # run algorithm
    if (diag_covariance) {
	f <- fit(y, X, groups, lambda, a0, b0, tau_a0, tau_b0, mu, s, g, diag_covariance,
	    track_elbo, track_elbo_every, track_elbo_mcn, niter, tol, verbose)
    } else {
	f <- fit(y, X, groups, lambda, a0, b0, tau_a0, tau_b0, mu, s, g, diag_covariance,
	    track_elbo, track_elbo_every, track_elbo_mcn, niter, tol, verbose)
	f$s <- lapply(f$S, function(s) matrix(s, nrow=sqrt(length(s))))
    }

    ta <- f$tau_a
    tb <- f$tau_b
   
    res <- list(
	mu = f$mu,
	s = f$s,
	g = f$g[!duplicated(groups)],
	beta_hat = f$mu * f$g,
	tau_hat = tb / (ta - 1),
	tau_a = ta,
	tau_b = tb,
	parameters = list(lambda = lambda, a0 = a0, b0=b0, tau_a0=tau_a0, tau_b0=tau_b0,
			  intercept=intercept, diag_covariance=diag_covariance,
			  groups=groups, model=1),
	converged = f$converged,
	iter = f$iter
    )
   
    if (track_elbo)
	res$elbo <- f$elbo

    return(res)
}

