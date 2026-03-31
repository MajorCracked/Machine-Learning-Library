#pragma once
#include "optimizer.h"
#include <map>

class Adam : public Optimizer {
    double lr, beta1, beta2, epsilon;
    std::map<Layer*, int> layer_t;
    std::map<Layer*, Matrix> mW, mb, vW, vb;

public:
    Adam(double lr=0.001, double beta1=0.9, double beta2=0.999, double epsilon=1e-8);
    void step(Layer* l) override;
    double get_lr() const override;
    void set_lr(double lr) override;
};