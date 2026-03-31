#include "bce_logits.h"
#include <cmath>

static double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double BCEWithLogits::loss(const Matrix& z, const Matrix& y) {
    double L = 0.0;
    for (int i = 0; i < z.rows; i++) {
        double zi = z(i,0);
        double yi = y(i,0);
        L += std::max(zi, 0.0) - zi * yi + std::log(1 + std::exp(-std::abs(zi)));
    }
    return L / z.rows;
}

Matrix BCEWithLogits::grad(const Matrix& z, const Matrix& y) {
    Matrix g(z.rows, 1);
    for (int i = 0; i < z.rows; i++) {
        double pi = sigmoid(z(i,0));
        g(i,0) = (pi - y(i,0)) / z.rows;
    }
    return g;
}
