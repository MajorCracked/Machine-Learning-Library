#pragma once
#include "../math/matrix.h"

class Accuracy {
public:
    double compute(const Matrix& pred,
                   const Matrix& target);
};