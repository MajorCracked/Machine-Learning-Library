#include "ce.h"
#include <cmath>
#include <stdexcept>

static double clamp(double x) {
    if (x < 1e-7) return 1e-7;
    if (x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

double CE::loss(const Matrix& pred, const Matrix& target) {
    if (pred.rows != target.rows || pred.cols != target.cols)
        throw std::runtime_error("CE shape mismatch");

    double sum = 0.0;
    for (int i = 0; i < pred.data.size(); i++) {
        double p = clamp(pred.data[i]);
        sum += -target.data[i] * std::log(p);
    }
    return sum / pred.rows;
}

Matrix CE::grad(const Matrix& pred, const Matrix& target) {
    Matrix g(pred.rows, pred.cols);
    for (int i = 0; i < pred.data.size(); i++) {
        g.data[i] = (pred.data[i] - target.data[i]) / pred.rows;
    }
    return g;
}
