#pragma once
#include "../model/model.h"
#include "../loss/loss.h"
#include "../optim/optimizer.h"
#include "../data/dataloader.h"
#include "../scheduler/lr_scheduler.h"

class Trainer {
    Model& model;
    Optimizer& optim;
    Loss* loss_ptr;
    LRScheduler* scheduler;

public:
    Trainer(Model& m,
            Optimizer& o,
            Loss* loss_ptr,
            LRScheduler* sched = nullptr);

    void fit(DataLoader& train_loader,
             DataLoader& val_loader,
             int epochs);

    void evaluate(DataLoader& loader);
};