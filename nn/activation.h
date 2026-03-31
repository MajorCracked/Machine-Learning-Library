#pragma once
#include "layer.h"

class ReLU : public Layer {
private:
    Matrix cache;

public:
    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad);
    void update(double lr);
};