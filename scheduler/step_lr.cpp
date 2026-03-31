#include "step_lr.h"

StepLR::StepLR(Optimizer& o,
               int step_size,
               double gamma)
    : LRScheduler(o),
      step_size(step_size),
      gamma(gamma) {}

void StepLR::step(int epoch) {

    if(epoch > 0 && epoch % step_size == 0) {
        optim.set_lr(optim.get_lr() * gamma);
    }
}