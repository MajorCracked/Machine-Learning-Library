#pragma once
#include "loss.h"

class BCE : public Loss {
public:
    double loss(const Matrix& pred,
                const Matrix& target) override;

    Matrix grad(const Matrix& pred,
                const Matrix& target) override;
};