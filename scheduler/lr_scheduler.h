#pragma once
#include "../optim/optimizer.h"

class LRScheduler {
protected:
    Optimizer& optim;

public:
    LRScheduler(Optimizer& o) : optim(o) {}
    virtual void step(int epoch) = 0;
    virtual ~LRScheduler() {}
};