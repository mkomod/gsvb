% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fit.r
\name{gsvb.fit}
\alias{gsvb.fit}
\title{Fit high-dimensional group-sparse regression models}
\usage{
gsvb.fit(
  y,
  X,
  groups,
  family = "gaussian",
  intercept = TRUE,
  diag_covariance = TRUE,
  lambda = 1,
  a0 = 1,
  b0 = length(unique(groups)),
  tau_a0 = 0.001,
  tau_b0 = 0.001,
  mu = NULL,
  s = apply(X, 2, function(x) 1/sqrt(sum(x^2) * tau_a0/tau_b0 + 2 * lambda)),
  g = rep(0.5, ncol(X)),
  track_elbo = TRUE,
  track_elbo_every = 5,
  track_elbo_mcn = 500,
  niter = 150,
  niter.refined = 20,
  tol = 0.001,
  verbose = TRUE,
  thresh = 0.02,
  l = 5,
  ordering = 0,
  init_method = "lasso"
)
}
\arguments{
\item{y}{response vector.}

\item{X}{input matrix.}

\item{groups}{group structure.}

\item{family}{which family and bound to use when fitting the model. One of:
\itemize{
    \item{\code{"gaussian"}}{ linear model with Gaussian noise}
    \item{\code{"bimomial-jensens"}}{ binomial family with logit link function where Jensen's inq. is used to upper bound the expected log-likelihood. Currently only supports a variational family with a diagonal covariance matrix.}
    \item{\code{"binomial-jaakkola"}}{ binomial family with logit link function where Jaakkola's bound for the logistic function. Currently supports variational families with both diagonal and group covariance matrices.}
    \item{\code{"binomial-refined"}}{ binomial family where a tighter bound for expected log-likelihood is used. *Note* can be slow. Currently only supports a variational family with diagonal covariance.}
    \item{\code{"poisson"}}{ poisson regression with log link function. Currently only supports variational familiy with a diagonal covariance matrix.}
}}

\item{intercept}{should an intercept term be included.}

\item{diag_covariance}{should a diagonal covariance matrix be used in the variational approximation. Note: if true then the *standard deviations* for each coefficient are returned. If false then covariance matrices for each group are returned.}

\item{lambda}{penalization hyperparameter for the multivariate exponential prior.}

\item{a0}{shape parameter for the Beta(a0, b0) mixing prior.}

\item{b0}{shape parameter for the Beta(a0, b0) mixing prior.}

\item{tau_a0}{shape parameter for the inverse-Gamma(a0, b0) prior on the variance, tau^2.}

\item{tau_b0}{scale parameter for the inverse-Gamma(a0, b0) prior on the variance, tau^2.}

\item{mu}{initial values of mu, the means of the variational family.}

\item{s}{initial values of s, the std. dev of the variational family.}

\item{g}{initial values of g, the group inclusion probabilities of the variational family.}

\item{track_elbo}{track the evidence lower bound (ELBO).}

\item{track_elbo_every}{the number of iterations between computing the ELBO.}

\item{track_elbo_mcn}{number of Monte-Carlo samples to compute the ELBO.}

\item{niter}{maximum number of iteration to run the algorithm for.}

\item{niter.refined}{maximum number of iteration to run the "binomial-refined" algorithm for.}

\item{tol}{convergence tolerance.}

\item{verbose}{print additional information.}

\item{thresh}{threshold used for the "logit-refined" family}

\item{l}{number of parameters used for the "logit-refined" family, samller is faster but more approximate.}

\item{ordering}{ordering of group updates. 0 is no ordering, 1 is random ordering, 2 is ordering by the norm of the group}

\item{init_method}{method to initialize the algorithm. One of:
\itemize{
    \item{\code{"lasso"}}{initialize using the group LASSO.}
    \item{\code{"random"}}{initialize using random values.}
    \item{\code{"ridge"}}{initialize using the ridge penalty.}
}}
}
\value{
The program output is a list containing:
\item{mu}{the means for the variational posterior.}
\item{s}{the std. dev. or covariance matrices for the variational posterior.}
\item{g}{the group inclusion probabilities.}
\item{beta_hat}{the variational posterior mean.}
\item{tau_hat}{the mean of the variance term. (linear only)}
\item{tau_a}{the shape parameter for the variational posterior of tau^2. This is an inverse-Gamma(tau_a0, tau_b0) distribution. (linear only)}
\item{tau_b}{the scale parameter for the variational posterior of tau^2. (linear only)}
\item{parameters}{a list containing the model hyperparameters and other model information}
\item{converged}{a boolean indicating if the algorithm has converged.}
\item{iter}{the number of iterations the algorithm was ran for.}
}
\description{
Fit high-dimensional group-sparse regression models
}
\section{Details}{
 TODO
}

\examples{
library(gsvb)

n <- 100
p <- 1000
gsize <- 5
groups <- c(rep(1:(p/gsize), each=gsize))

X <- matrix(rnorm(n * p), nrow=n, ncol=p)
b <- c(rep(0, gsize), rep(-4, gsize), rep(8, gsize), rep(0, p - 3 * gsize))
y <- X \%*\% b + rnorm(n, 0, 1)

f <- gsvb.fit(y, X, groups)

plot(f$beta_hat, col=4, ylab=expression(hat(beta)))
points(b, pch=20)


}
