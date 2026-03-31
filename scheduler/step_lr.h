#pragma once
#include "lr_scheduler.h"

class StepLR : public LRScheduler {
    int step_size;
    double gamma;

public:
    StepLR(Optimizer& o,
           int step_size,
           double gamma);

    void step(int epoch) override;
};