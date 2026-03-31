#pragma once
#include "../math/matrix.h"

class Layer {
public:
    virtual Matrix forward(const Matrix& x) = 0;
    virtual Matrix backward(const Matrix& grad) = 0;
    virtual void update(double lr) = 0;
    virtual ~Layer() {}
};