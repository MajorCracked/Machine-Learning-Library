#include "sigmoid.h"
#include <cmath>

Matrix Sigmoid::forward(const Matrix& x) {
    cache = Matrix(x.rows, x.cols);

    for (int i = 0; i < x.data.size(); i++) {
        cache.data[i] = 1.0 / (1.0 + std::exp(-x.data[i]));
    }

    return cache;
}

Matrix Sigmoid::backward(const Matrix& grad) {
    Matrix g(grad.rows, grad.cols);
    for (int i = 0; i < grad.data.size(); i++) {
        g.data[i] = grad.data[i] * cache.data[i] * (1.0 - cache.data[i]);
    }
    return g;
}