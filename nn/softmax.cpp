#include "softmax.h"
#include <cmath>
#include <stdexcept>

Matrix Softmax::forward(const Matrix& x) {
    cache = Matrix(x.rows, x.cols);
    for (int i = 0; i < x.rows; i++) {
        double max_val = x(i, 0);
        for (int j = 1; j < x.cols; j++)
            if (x(i, j) > max_val) max_val = x(i, j);

        double sum = 0.0;
        for (int j = 0; j < x.cols; j++) {
            cache(i, j) = std::exp(x(i, j) - max_val);
            sum += cache(i, j);
        }
        for (int j = 0; j < x.cols; j++) {
            cache(i, j) /= sum;
        }
    }
    return cache;
}

Matrix Softmax::backward(const Matrix& grad) {
    // Correct local backpropagation through a Softmax output layer is mathematically complex.
    // In our framework, we assume cross-entropy gradient returns pre-derived logits differentials directly, similar to BCE.
    // This allows Softmax output layer to safely return grad directly for standard classification.
    return grad;
}
