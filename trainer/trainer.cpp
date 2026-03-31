#include "trainer.h"
#include <iostream>

Trainer::Trainer(Model& m,
                 Optimizer& o,
                 Loss* loss_ptr,
                 LRScheduler* sched)
    : model(m), optim(o), loss_ptr(loss_ptr), scheduler(sched) {}

void Trainer::fit(DataLoader& train_loader,
                  DataLoader& val_loader,
                  int epochs) {

    for(int e=0; e<epochs; e++) {

        train_loader.reset(true);
        double train_loss = 0;
        int batches = 0;

        while(train_loader.has_next()) {
            Matrix Xb, Yb;
            train_loader.next(Xb, Yb);

            Matrix pred = model.forward(Xb);
            train_loss += loss_ptr->loss(pred, Yb);

            Matrix g = loss_ptr->grad(pred, Yb);
            model.backward(g);
            model.step(optim);

            batches++;
        }

        train_loss /= batches;

        if (scheduler) {
            scheduler->step(e);
        }
        int print_freq = (epochs >= 10) ? (epochs / 10) : 1;
        if(e % print_freq == 0 || e == epochs-1) {
            std::cout << "[TRAIN] epoch "
                      << e
                      << " loss "
                      << train_loss
                      << std::endl;

            evaluate(val_loader);
        }
    }
}

void Trainer::evaluate(DataLoader& loader) {

    loader.reset(false);

    double total_loss = 0;
    int batches = 0;

    while(loader.has_next()) {
        Matrix Xb, Yb;
        loader.next(Xb, Yb);

        Matrix pred = model.forward(Xb);

        total_loss += loss_ptr->loss(pred, Yb);
        batches++;
    }

    std::cout << "[VAL]   loss "
              << total_loss / batches
              << std::endl;
}