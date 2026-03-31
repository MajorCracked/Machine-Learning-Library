#pragma once
#include "layer.h"

class Softmax : public Layer {
    Matrix cache;
public:
    Matrix forward(const Matrix& x) override;
    Matrix backward(const Matrix& grad) override;
    void update(double lr) override {}
};
