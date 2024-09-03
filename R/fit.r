#' Fit high-dimensional group-sparse regression models
#'
#' @param y response vector.
#' @param X input matrix.
#' @param groups group structure.
#' @param family which family and bound to use when fitting the model. One of:
#' \itemize{
#' 	\item{\code{"gaussian"}}{ linear model with Gaussian noise}
#' 	\item{\code{"bimomial-jensens"}}{ binomial family with logit link function where Jensen's inq. is used to upper bound the expected log-likelihood. Currently only supports a variational family with a diagonal covariance matrix.}
#' 	\item{\code{"binomial-jaakkola"}}{ binomial family with logit link function where Jaakkola's bound for the logistic function. Currently supports variational families with both diagonal and group covariance matrices.}
#' 	\item{\code{"binomial-refined"}}{ binomial family where a tighter bound for expected log-likelihood is used. *Note* can be slow. Currently only supports a variational family with diagonal covariance.}
#' 	\item{\code{"poisson"}}{ poisson regression with log link function. Currently only supports variational familiy with a diagonal covariance matrix.}
#' }
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
#' @param niter.refined maximum number of iteration to run the "binomial-refined" algorithm for.
#' @param tol convergence tolerance.
#' @param verbose print additional information.
#' @param thresh threshold used for the "logit-refined" family
#' @param l number of parameters used for the "logit-refined" family, samller is faster but more approximate.
#' @param ordering ordering of group updates. 0 is no ordering, 1 is random ordering, 2 is ordering by the norm of the group
#' @param init_method method to initialize the algorithm. One of:
#' \itemize{
#' 	\item{\code{"lasso"}}{initialize using the group LASSO.}
#' 	\item{\code{"random"}}{initialize using random values.}
#' }
#' 
#' 
#' @return The program output is a list containing:
#' \item{mu}{the means for the variational posterior.}
#' \item{s}{the std. dev. or covariance matrices for the variational posterior.}
#' \item{g}{the group inclusion probabilities.}
#' \item{beta_hat}{the variational posterior mean.}
#' \item{tau_hat}{the mean of the variance term. (linear only)}
#' \item{tau_a}{the shape parameter for the variational posterior of tau^2. This is an inverse-Gamma(tau_a0, tau_b0) distribution. (linear only)}
#' \item{tau_b}{the scale parameter for the variational posterior of tau^2. (linear only)}
#' \item{parameters}{a list containing the model hyperparameters and other model information}
#' \item{converged}{a boolean indicating if the algorithm has converged.}
#' \item{iter}{the number of iterations the algorithm was ran for.}
#' 
#' @section Details: TODO
#'
#' @examples
#' library(gsvb)
#'
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
gsvb.fit <- function(y, X, groups, family="gaussian", intercept=TRUE, 
    diag_covariance=TRUE, lambda=1, a0=1, b0=length(unique(groups)), 
    tau_a0=1e-3, tau_b0=1e-3, mu=NULL, 
    s=apply(X, 2, function(x) 1/sqrt(sum(x^2)*tau_a0/tau_b0+2*lambda)),
    g=rep(0.5, ncol(X)), track_elbo=TRUE, track_elbo_every=5, 
    track_elbo_mcn=5e2, niter=150, niter.refined=20, 
    tol=1e-3, verbose=TRUE, thresh=0.02, l=5, ordering=0, init_method="lasso") 
{
    family <- pmatch(family, c("gaussian", "binomial-jensens", "binomial-jaakkola", 
	    "binomial-refined", "poisson"))

	if (is.null(init_method)) init_method <- "lasso"
	init_method <- pmatch(init_method, c("lasso", "random"))

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
    if (is.na(family))
	stop("Invalid family")
    if (any(family == c(2,3,4)) && !all(y == 1 | y == 0))
	stop("Classification requires y to be in {0, 1}")

    # init s for other families
    if (any(family == c(2,3,4,5)) && (is.null(tau_a0) || is.null(tau_b0))) {
	tau_a0 <- tau_b0 <- 1
	s <- apply(X, 2, function(x) 1/sqrt(sum(x^2)+2*lambda))
    }

    # pre-processing
    if (intercept) {
	groups <- c(min(groups) - 1, groups) + 1
	X <- cbind(rep(1, nrow(X)), X)
	n <- nrow(X)
	
	# update the initial parameters
	if (length(s) == ncol(X) - 1)
	    s <- c(1/sqrt(sqrt(n) * tau_a0 / tau_b0 + 2 *lambda), s)

	if (length(g) != ncol(X))
	    g <- c(0.5, g)
    }

    # initialize using the group LASSO
	if (is.null(mu))
	{
		if (init_method == 1)
		{
		# Note: the intercept is handled by adding a column of 1s to the
		# design matrix X and is therefore disabled for gglasso
		if (family == 1) {
			glfit <- gglasso::gglasso(X, y, groups, nlambda=10, intercept=FALSE)
		} else if (any(c(2,3,4) == family)) {
			yy <- y
			yy[which(y == 0)] <- -1
			glfit <- gglasso::gglasso(X, yy, groups, loss="logit", nlambda=10, 
			intercept=FALSE)
		} else if (5 == family) {
			message("Group LASSO not available for the poisson model. Using the LASSO to initialize.")
			glfit <- glmnet::glmnet(X, y, family="poisson", standardize=FALSE, 
			intercept=FALSE)
		}

		# take mu as the estimate for smallest reg parameter
		mu <- glfit$beta[ , length(glfit$lambda)]
		}
		else if (init_method == 2)
		{
			mu <- rnorm(ncol(X), 0, 0.5)
		}
	}

    if (family == 1) # LINEAR
    {
	f <- fit_linear(y, X, groups, lambda, a0, b0, tau_a0, tau_b0, 
	    mu, s, g, diag_covariance, track_elbo, track_elbo_every, 
	    track_elbo_mcn, niter, tol, verbose, ordering)
    }
    if (family == 2) # LOGISTIC - JENSEN BOUND
    {
	message("Logistic model with Jensen's bound currently only supoorts a variational family with diagonal covariance matrix.")

	diag_covariance <- TRUE
	f <- fit_logistic(y, X, groups, lambda, a0, b0,
	    mu, s, g, diag_covariance, track_elbo, track_elbo_every,
	    track_elbo_mcn, thresh, l, niter, 2, tol, verbose)
    }
    if (family == 3) # LOGISTIC - JAAKKOLA BOUND
    {
	f <- fit_logistic(y, X, groups, lambda, a0, b0,
	    mu, s, g, diag_covariance, track_elbo, track_elbo_every,
	    track_elbo_mcn, thresh, l, niter, 3, tol, verbose)
    }
    if (family == 4) # LOGISTIC - OUR BOUND
    {
	message("Logistic model with new bound currently only supoorts a variational family with diagonal covariance matrix.")
	diag_covariance <- TRUE

	f <- fit_logistic(y, X, groups, lambda, a0, b0,
	    mu, s, g, diag_covariance, FALSE, track_elbo_every,
	    track_elbo_mcn, thresh, l, niter, 3, tol, verbose)

	# if mu is provided by the user then this input is taken
	# and refined with our tight upper bound.
	f <- fit_logistic(y, X, groups, lambda, a0, b0,
	    f$mu, f$s, f$g, diag_covariance, track_elbo, track_elbo_every,
	    track_elbo_mcn, thresh, l, niter.refined, 1, tol, verbose)
    }
    if (family == 5) # POISSON REG
    {
	f <- fit_poisson(y, X, groups, lambda, a0, b0, mu, s, g,
	    diag_covariance, track_elbo, track_elbo_every, track_elbo_mcn, 
	    niter, tol, verbose)
    }
    
    if (diag_covariance == FALSE && any(c(1,3,5) == family)) {
	f$s <- lapply(f$S, function(s) matrix(s, nrow=sqrt(length(s))))
    }
   
    res <- list(
	mu = f$mu,
	s = f$s,
	g = f$g[!duplicated(groups)],
	beta_hat = f$mu * f$g,
	parameters = list(lambda = lambda, a0 = a0, b0=b0,
			  intercept=intercept, diag_covariance=diag_covariance,
			  groups=groups, family=family),
	converged = f$converged,
	iter = f$iter
    )

    if (family == 1) {
	res$tau_a = f$tau_a
	res$tau_b = f$tau_b
	res$tau_hat = f$tau_b / (f$tau_a - 1)
	res$parameters$tau_a0=tau_a0
	res$parameters$tau_b0=tau_b0
    }
   
    if (track_elbo)
	res$elbo <- f$elbo

    return(res)
}

