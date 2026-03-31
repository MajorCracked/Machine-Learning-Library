#pragma once
#include "layer.h"
#include <fstream>

class Dense : public Layer {
private:
    Matrix W, b;
    Matrix input;
    Matrix dW, db;

public:
    Dense(int in, int out);

    Matrix forward(const Matrix& x);
    Matrix backward(const Matrix& grad);
    void update(double lr);
    void save(std::ofstream& f);
    void load(std::ifstream& f);

    Matrix& weights();
    Matrix& bias();
    Matrix& gradW();
    Matrix& gradB();
};