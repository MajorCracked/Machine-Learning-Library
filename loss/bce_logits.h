#pragma once
#include "loss.h"

class BCEWithLogits : public Loss {
public:
    double loss(const Matrix& logits, const Matrix& y) override;
    Matrix grad(const Matrix& logits, const Matrix& y) override;
};
