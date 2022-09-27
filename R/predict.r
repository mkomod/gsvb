#' Sample from the posterior predictive distribution
#'
#' @param fit the fit model.
#' @param newdata input feature matrix.
#' @param groups group structure.
#' @param mcn number of Monte-Carlo samples.
#' @param quantiles quantiles to return 
#' @param return_samples return all the samples
#'
#' @return a list with the mean of the posterior predictive, the quantiles, and optionally the samples.
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
#' gsvb.predict(f, groups, X) 
#'
#' @export
gsvb.predict <- function(fit, groups, newdata, mcn=1e4, 
    quantiles=c(0.025, 0.975), return_samples=FALSE) 
{
    if (fit$parameters$intercept) {
	newdata <- cbind(1, newdata)
	groups <- c(1, groups + 1)
    }

    M <- length(fit$g)
    n <- nrow(newdata)
    sigma <- sqrt(fit$tau_b / fit$tau_a)

    y.star <- replicate(mcn, 
    {
	G <- runif(M) <= fit$g
	grp <- G[groups]

	if (length(G) == 0)
	    return(sigma * rt(n, 2 * fit$tau_a))

	if (fit$parameters$diag_covariance)
	{
	    mu <- rnorm(sum(grp), fit$mu[grp], fit$s[grp])
	} else 
	{
	    mu <- sapply(which(G), function(j) {
		Gj <- which(groups == j)
		rnorm(length(Gj)) %*% t(chol(f$s[[j]])) + f$m[Gj]
	    })
	    mu <- matrix(as.numeric(unlist(mu)), ncol=1)
	}

	newdata[ , grp] %*% mu + sigma * rt(n, 2 * fit$tau_a)
    }, simplify="matrix")

    y.star <- matrix(y.star, nrow=n)

    res <- list(
	mean=apply(y.star, 1, mean),
	quantiles=apply(y.star, 1, quantile, probs=quantiles)
    )

    if (return_samples) {
	res <- c(res, list(samples=y.star))
    }

    return(res)
}
