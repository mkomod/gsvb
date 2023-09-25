# Group sparse Variational Bayes

GSVB is software for scalable group sparse regression. Unlike other state-of-the-art group selection methods, GSVB provides uncertainty quantification at scale. Offering an alternative that is as fast as maximum a posteriori methods, and orders of magnitude faster than MCMC.

Currenlty GSVB is available for the linear, logistic and Poisson models. 

## Install

```
devtools::install_github("mkomod/gsvb")
```

## Example


```{R}
library(gsvb)

n <- 100
p <- 1000
gsize <- 5
groups <- c(rep(1:(p/gsize), each=gsize))

X <- matrix(rnorm(n * p), nrow=n, ncol=p)
b <- c(rep(0, gsize), rep(-4, gsize), rep(8, gsize), rep(0, p - 3 * gsize))
y <- X %*% b + rnorm(n, 0, 1)

f <- gsvb.fit(y, X, groups)

plot(f$beta_hat, col=4, ylab=expression(hat(beta)))
points(b, pch=20)
```


## Details

GSVB computes a variational approximation the full group sprase posterior. The prior used for the model coefficients is
```math
\begin{align}
\beta_j &\ \sim z_j \Psi(\beta_j, \lambda) + (1-z_j) \dellta_0 \\
z_j &\ \sim \text{Bern)(\theta_j) \\
\theta_j &\ \sim \text{Beta}(a_0, b_0)
\end{align}
```
where $Psi(\beta, \lambda)$ is the multivariate double exponential distribution and $\dellta_0$ is tthe multivariate dirac mass. 

Under this prior sparisity is imposed via the Dirac mass which sets the entire group to zero when $\_j = 0$ and enables the group to be non-zero when $z_j = 1$. Exploring the entire model space is not feasible for a large number of groups. Therefore, we approximate the posterior through a variational approximation.

Full details are available at https://arxiv.org/abs/2309.10378


