#include "utils.h"

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}


vec sigmoid(const vec &x) {
    return 1/(1 + exp(-x));
}


vec mvnMGF(const mat &X, const vec &mu, const vec &sig) 
{
    return exp(X * mu + 0.5 * (X % X) * (sig % sig));
}


