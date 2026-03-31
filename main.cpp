#include "model/model.h"
#include "nn/dense.h"
#include "nn/activation.h"
#include "loss/mse.h"
#include "optim/adam.h"
#include "trainer/trainer.h"
#include "data/dataset.h"
#include "data/dataloader.h"
#include <iostream>
#include <memory>
#include <cmath>

using namespace std;

int main() {
    cout << "Loading F1 Synthetic Dataset (200k rows)..." << endl;
    Dataset raw_data = Dataset::load_csv("data/F1_2026_Synthetic_Dataset_200k.csv", 1);
    cout << "Data loaded! X shape: " << raw_data.X.rows << "x" << raw_data.X.cols << endl;

    // Normalizing X features (Min-Max Scaling)
    for (int j = 0; j < raw_data.X.cols; j++) {
        double min_val = raw_data.X.data[j];
        double max_val = raw_data.X.data[j];
        for (int i = 0; i < raw_data.X.rows; i++) {
            double v = raw_data.X.data[i * raw_data.X.cols + j];
            if (v < min_val) min_val = v;
            if (v > max_val) max_val = v;
        }
        for (int i = 0; i < raw_data.X.rows; i++) {
            if (max_val - min_val > 1e-7)
                raw_data.X.data[i * raw_data.X.cols + j] = (raw_data.X.data[i * raw_data.X.cols + j] - min_val) / (max_val - min_val);
        }
    }

    // Normalizing Y scalar
    double max_y = 1e-7;
    for(int i = 0; i < raw_data.Y.rows; i++) {
        if (std::abs(raw_data.Y.data[i]) > max_y) max_y = std::abs(raw_data.Y.data[i]);
    }
    for(int i = 0; i < raw_data.Y.rows; i++) {
        raw_data.Y.data[i] /= max_y;
    }
    cout << "Data normalized! (Max Y scaling factor: " << max_y << ")" << endl;

    DataLoader loader(raw_data, 128); // Standard robust batch chunk

    Model net;
    net.add(make_unique<Dense>(10, 64));
    net.add(make_unique<ReLU>());
    net.add(make_unique<Dense>(64, 32));
    net.add(make_unique<ReLU>());
    net.add(make_unique<Dense>(32, 1)); 

    MSE mse_loss;
    Adam adam(0.001);

    Trainer trainer(net, adam, &mse_loss);

    cout << "Starting Deep MLP Regression Training on 200K Samples..." << endl;
    try {
        trainer.fit(loader, loader, 5); 
    } catch(const std::exception& e) {
        cerr << "CRITICAL ERROR: " << e.what() << endl;
    } catch(...) {
        cerr << "UNKNOWN CRASH" << endl;
    }

    cout << "\nSaving trained weights to f1_model.bin..." << endl;
    net.save("f1_model.bin");

    cout << "Training Finished. Press Enter to exit.";
    cin.get();
    return 0;
}
