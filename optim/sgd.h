#pragma once
#include "optimizer.h"

class SGD : public Optimizer {
    double lr;
public:
    explicit SGD(double lr);

    void step(Layer* l) override;   // ✅ POINTER (matches base)
    double get_lr() const override;
    void set_lr(double lr) override;
};