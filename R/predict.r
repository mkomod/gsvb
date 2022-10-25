#' Sample from the posterior predictive distribution
#'
#' @param fit the fit model.
#' @param newdata input feature matrix.
#' @param samples number of Monte-Carlo samples.
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
gsvb.predict <- function(fit, newdata, samples=1e4, 
    quantiles=c(0.025, 0.975), return_samples=FALSE) 
{
    M <- length(fit$g)
    n <- nrow(newdata)
    groups <- fit$parameters$groups

    if (fit$parameters$intercept)
	newdata <- cbind(1, newdata)

    # samples <- gsvb::gsvb.sample(fit, samples=samples)
    samples <- gsvb.sample(fit, samples=samples)
    Xb <- newdata %*% samples$beta

    if (fit$parameters$family == 1)
    {
	sigma <- sqrt(fit$tau_b / fit$tau_a)
	y.star <- Xb + sigma * rt(prod(dim(Xb)), 2 + fit$tau_a)
    }
    else if(any(fit$parameters$family == c(2,3,4)))
    {
	y.star <- 1/(1 + exp(-Xb))	
    }
    else if(fit$parameters$family == 5)
    {
	lambda <- exp(Xb)
	y.star <- matrix(rpois(prod(dim(lambda)), lambda), nrow=n)
    }

    res <- list(
	mean=apply(y.star, 1, mean),
	quantiles=apply(y.star, 1, quantile, probs=quantiles)
    )

    if (return_samples) {
	res <- c(res, list(samples=y.star))
    }

    return(res)
}
