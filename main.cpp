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
#include <string>

using namespace std;

int main() {
    cout << "========================================" << endl;
    cout << "   UNIVERSAL C++ DEEP LEARNING ENGINE   " << endl;
    cout << "========================================" << endl;

    string dataset_path;
    int label_cols;
    int epochs;
    string output_model_name;

    cout << "Enter path to CSV dataset (e.g., data/dataset.csv): ";
    cin >> dataset_path;

    cout << "Enter number of target labels at the end of CSV: ";
    cin >> label_cols;

    cout << "Enter number of training epochs: ";
    cin >> epochs;

    cout << "Enter filename to save trained model (e.g., model.bin): ";
    cin >> output_model_name;

    cout << "\nLoading Dataset..." << endl;
    Dataset raw_data(Matrix(0,0), Matrix(0,0));
    try {
        raw_data = Dataset::load_csv(dataset_path, label_cols);
    } catch(const exception& e) {
        cerr << "Failed to load CSV: " << e.what() << endl;
        return 1;
    }
    cout << "Data loaded! Training Features: " << raw_data.X.cols << ", Targets: " << raw_data.Y.cols << endl;
    cout << "Total Samples: " << raw_data.X.rows << endl;

    cout << "Normalizing features..." << endl;
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

    double max_y = 1e-7;
    for(int i = 0; i < raw_data.Y.rows; i++) {
        if (std::abs(raw_data.Y.data[i]) > max_y) max_y = std::abs(raw_data.Y.data[i]);
    }
    for(int i = 0; i < raw_data.Y.rows; i++) {
        raw_data.Y.data[i] /= max_y;
    }

    DataLoader loader(raw_data, 128); // Standard robust batch chunk

    // Dynamically builds architecture exactly mirroring CSV columns!
    Model net;
    int in_f = raw_data.X.cols;
    int out_f = raw_data.Y.cols;

    net.add(make_unique<Dense>(in_f, 64));
    net.add(make_unique<ReLU>());
    net.add(make_unique<Dense>(64, 32));
    net.add(make_unique<ReLU>());
    net.add(make_unique<Dense>(32, out_f));

    MSE mse_loss;
    Adam adam(0.001);
    Trainer trainer(net, adam, &mse_loss);

    cout << "\nStarting Native C++ Threaded Training..." << endl;
    try {
        trainer.fit(loader, loader, epochs); 
    } catch(const std::exception& e) {
        cerr << "CRITICAL ERROR: " << e.what() << endl;
    } catch(...) {
        cerr << "UNKNOWN CRASH" << endl;
    }

    cout << "\nSaving trained weights to " << output_model_name << "..." << endl;
    net.save(output_model_name);

    cout << "Training Finished! Press Enter to exit.";
    cin.ignore(10000, '\n');
    cin.get();
    
    return 0;
}
