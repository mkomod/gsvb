#' Sample from the variational posterior distribution
#'
#' @param fit the fit model.
#' @param samples number of samples
#'
#' @return a list containing:
#' \item{beta}{a matrix of samples from beta}
#' \item{tau}{vector of samples from tau (only for linear model)}
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
#' s <- gsvb.sample(f)
#' hist(s$beta[10, ])
#'
#' @export
gsvb.sample <- function(fit, samples=1e4)
{
    groups <- fit$parameters$groups
    M <- length(fit$g)

    beta <- replicate(samples, 
    {
	active_groups <- runif(M) <= fit$g
	grp <- active_groups[groups]

	b <- rep(0, length(grp))

	if (length(active_groups) == 0)
	    return(b)

	if (fit$parameters$diag_covariance)
	{
	    # sample only from the active groups
	    m <- rnorm(sum(grp), fit$mu[grp], fit$s[grp])
	} else 
	{
	    m <- sapply(which(active_groups), function(j) {
		Gj <- which(groups == j)
		rnorm(length(Gj)) %*% t(chol(fit$s[[j]])) + fit$m[Gj]
	    })
	    m <- matrix(as.numeric(unlist(m)), ncol=1)
	}
	
	if (length(grp) != 0) {
	    b[grp] <- m
	}
	return(b)
    }, simplify="matrix")

    if (fit$parameters$family == 1) {
	tau <- 1/rgamma(samples, shape=fit$tau_a, rate=fit$tau_b)
	return(list(beta=beta, tau=tau))
    }

    return(list(beta=beta))
}
