#pragma once
#include "../math/matrix.h"

class Loss {
public:
    virtual double loss(const Matrix& pred, const Matrix& target) = 0;
    virtual Matrix grad(const Matrix& pred, const Matrix& target) = 0;
    virtual ~Loss() = default;
};
