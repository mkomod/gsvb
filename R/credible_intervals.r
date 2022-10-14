#' Construct the marginal credible intervals for the model coefficients
#'
#' @param fit fit model
#' @param prob the probability contained in the credible interval of each coefficient
#'
#' @return matrix with the columns for:
#' \item{lower}{the lower bound of the credible interval}
#' \item{upper}{the upper bound of the credible interval}
#' \item{contains.dirac}{indicates whether the Dirac mass at zero is contained in the interval}
#' 
#' @section Details: 
#' Returns the interval of highest posterior density containing prob * 100% of the mass
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
#' credible.intervals(f, prob=0.95)
#' 
#' @export
gsvb.credible_intervals <- function(fit, prob=0.95)
{
    a <- 1 - prob
    
    # get the std. dev. for the marginal
    if (!fit$parameters$diag_covariance) {
	s <- sqrt(unlist(sapply(fit$s, function(s) diag(s))))
    } else {
	s <- fit$s
    }
    
    g <- fit$g[fit$parameters$groups]

    credible.interval <- sapply(1:length(g), function(i) 
    {
	g <- g[i]
	m <- fit$mu[i]
	s <- s[i]
	
	if (g > 1 - a) 
	{
	    # compute the interval that contains 1 - a.g of the mass
	    # i.e. if g = 0.97 then the interval needs to be wider to
	    # contain 95% of the total mass
	    a.g <- 1 - (1-a)/g
	    interval <- qnorm(c(a.g/2, 1-a.g/2), m, s)
	    contains.dirac <- FALSE
	    
	    if (interval[1] <= 0 && interval[2] >= 0) {
		# if interval contains Dirac mass it needs to be smaller
		interval <- qnorm(c(a.g/2 + (1-g)/2, 1 - a.g/2 - (1-g)/2), m, s)
		contains.dirac <- TRUE
	    }

	    return(c(lower=interval[1], upper=interval[2], 
		     contains.dirac=contains.dirac))
	} else if (g < a) 
	{
	    # if the spike contains (1-a)% of the mass then we take 
	    # the Dirac mass at 0
	    return(c(lower=0, upper=0, contains.dirac=T))
	} else 
	{
	    # will always contain the Dirac
	    # so we remove the density accounted for by the
	    # Dirac mass from the interval
	    interval <- qnorm(c(a/2 + (1-g)/2, 1-a/2 -(1-g)/2), m, s)

	    return(c(lower=interval[1], upper=interval[2], 
		    contains.dirac=TRUE))
	}
    })

    return(t(credible.interval))
}

