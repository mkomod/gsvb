% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/elbo.r
\name{gsvb.elbo}
\alias{gsvb.elbo}
\title{Compute the Evidence Lower Bound (ELBO)}
\usage{
gsvb.elbo(fit, y, X, mcn = 500, approx = FALSE, approx_thresh = 0.001)
}
\arguments{
\item{fit}{the fit model.}

\item{y}{response vector.}

\item{X}{input matrix.}

\item{mcn}{number of Monte-Carlo samples.}

\item{approx}{elements of gamma less than an approximation threshold are not used in computations.}

\item{approx_thresh}{the threshold below which elements of gamma are not used.}
}
\value{
the ELBO (numeric)
}
\description{
Compute the Evidence Lower Bound (ELBO)
}
\section{Details}{
 TODO
}

\examples{
n <- 100
p <- 1000
gsize <- 5
groups <- c(rep(1:(p/gsize), each=gsize))

X <- matrix(rnorm(n * p), nrow=n, ncol=p)
b <- c(rep(0, gsize), rep(-4, gsize), rep(8, gsize), rep(0, p - 3 * gsize))
y <- X \%*\% b + rnorm(n, 0, 1)

f <- gsvb.fit(y, X, groups)
gsvb.elbo(f, y, X, groups) 

}
