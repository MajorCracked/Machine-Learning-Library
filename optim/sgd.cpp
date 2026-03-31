#include "sgd.h"

SGD::SGD(double lr) : lr(lr) {}

void SGD::step(Layer* l) {
    l->update(lr);
}

double SGD::get_lr() const {
    return lr;
}

void SGD::set_lr(double lr_) {
    lr = lr_;
}