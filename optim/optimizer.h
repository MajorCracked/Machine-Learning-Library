#pragma once
#include "../nn/layer.h"

class Optimizer {
public:
    virtual void step(Layer* l) = 0;
    virtual ~Optimizer() {}
    virtual double get_lr() const = 0;
    virtual void set_lr(double lr) = 0;
};