#include "mse.h"
#include <stdexcept>
#include <cmath>

double MSE::loss(const Matrix& pred, const Matrix& target) {
    if (pred.rows != target.rows || pred.cols != target.cols)
        throw std::runtime_error("MSE shape mismatch");

    double sum = 0.0;
    for(int i=0;i<pred.data.size();i++) {
        double diff = pred.data[i] - target.data[i];
        sum += diff * diff;
    }
    return sum / pred.rows;
}

Matrix MSE::grad(const Matrix& pred, const Matrix& target) {
    Matrix g(pred.rows, pred.cols);
    for(int i=0;i<pred.data.size();i++) {
        g.data[i] = 2.0 * (pred.data[i] - target.data[i]) / pred.rows;
    }
    return g;
}