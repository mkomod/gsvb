# Group sparse Variational Bayes

## Details

https://arxiv.org/abs/2309.10378


## Install

```
devtools::install_github("mkomod/gsvb")
```

## Example


```
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


