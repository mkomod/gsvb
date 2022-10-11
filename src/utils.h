#ifndef GSVB_UTILS_H
#define GSVB_UTILS_H

#include "RcppEnsmallen.h"

#include "gsvb_types.h"

double sigmoid(double x);
vec sigmoid(const vec &x);

vec mvnMGF(const mat &X, const vec &mu, const vec &sig);

#endif
