#pragma once
#include "layer.h"

class Sigmoid : public Layer {
    Matrix cache;
public:
    Matrix forward(const Matrix& x) override;
    Matrix backward(const Matrix& grad) override;
    void update(double) override {}
};