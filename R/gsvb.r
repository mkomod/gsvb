
#
gsvb.fit <- function(y, X, groups, a0, b0, lambda) 
{
    # pre-processing
    group.order <- order(groups)
    groups <- groups[group.order]
    X <- X[ , group.order]
    
    # vars
    n <- nrow(X)
    p <- ncol(X)
    
    # initialize parameters
    mu <- rnorm(p)
    s <- rep(0.1, p)
    g <- rep(0.1, p)

    # run algorithm

   

    # re-order to match input order
    mu <- mu[group.order]
    s <- s[group.order]
    g <- g[group.order]
   
    
    res <- list(
	mu = mu,
	s = s,
	g = g,
	beta_hat = mu * g
    )

    return(res)
}
