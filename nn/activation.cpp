#include "activation.h"

Matrix ReLU::forward(const Matrix& x) {
    cache = x;
    Matrix r = x;
    for(auto& v : r.data)
        v = v > 0 ? v : 0;
    return r;
}

Matrix ReLU::backward(const Matrix& grad) {
    Matrix g = grad;
    for(int i=0;i<g.data.size();i++)
        g.data[i] = cache.data[i] > 0 ? g.data[i] : 0;
    return g;
}

void ReLU::update(double lr) {}