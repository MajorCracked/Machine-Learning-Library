#include "bce.h"
#include <cmath>
#include <stdexcept>

static double clamp(double x) {
    if(x < 1e-7) return 1e-7;
    if(x > 1 - 1e-7) return 1 - 1e-7;
    return x;
}

double BCE::loss(const Matrix& pred,
                 const Matrix& target) {

    if(pred.rows != target.rows || pred.cols != target.cols)
        throw std::runtime_error("BCE shape mismatch");

    double sum = 0.0;

    for(int i=0;i<pred.data.size();i++) {
        double p = clamp(pred.data[i]);
        double y = target.data[i];

        sum += -(y * log(p) + (1 - y) * log(1 - p));
    }

    return sum / pred.data.size();
}

Matrix BCE::grad(const Matrix& pred,
                 const Matrix& target) {

    Matrix g(pred.rows, pred.cols);

    for(int i=0;i<pred.data.size();i++) {
        double p = clamp(pred.data[i]);
        double y = target.data[i];

        double denom = std::max(p * (1.0 - p), 1e-7);
        g.data[i] = (p - y) / denom;
    }

    return g;
}