#include "adam.h"
#include "../nn/dense.h"
#include <cmath>

Adam::Adam(double lr, double beta1, double beta2, double epsilon)
    : lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

void Adam::step(Layer* l) {
    Dense* d = dynamic_cast<Dense*>(l);
    if (!d) {
        l->update(lr);
        return;
    }

    if (mW.find(l) == mW.end()) {
        mW[l] = Matrix::zeros(d->weights().rows, d->weights().cols);
        vW[l] = Matrix::zeros(d->weights().rows, d->weights().cols);
        mb[l] = Matrix::zeros(d->bias().rows, d->bias().cols);
        vb[l] = Matrix::zeros(d->bias().rows, d->bias().cols);
        layer_t[l] = 0;
    }

    layer_t[l]++;
    int t = layer_t[l];

    Matrix& W = d->weights();
    Matrix& b = d->bias();
    Matrix& dW = d->gradW();
    Matrix& db = d->gradB();

    for(int i=0; i<W.data.size(); i++) {
        mW[l].data[i] = beta1 * mW[l].data[i] + (1 - beta1) * dW.data[i];
        vW[l].data[i] = beta2 * vW[l].data[i] + (1 - beta2) * dW.data[i] * dW.data[i];
        double m_hat = mW[l].data[i] / (1 - std::pow(beta1, t));
        double v_hat = vW[l].data[i] / (1 - std::pow(beta2, t));
        W.data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }

    for(int i=0; i<b.data.size(); i++) {
        mb[l].data[i] = beta1 * mb[l].data[i] + (1 - beta1) * db.data[i];
        vb[l].data[i] = beta2 * vb[l].data[i] + (1 - beta2) * db.data[i] * db.data[i];
        double m_hat = mb[l].data[i] / (1 - std::pow(beta1, t));
        double v_hat = vb[l].data[i] / (1 - std::pow(beta2, t));
        b.data[i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

double Adam::get_lr() const { return lr; }
void Adam::set_lr(double lr_) { lr = lr_; }