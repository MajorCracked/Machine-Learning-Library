#include "dense.h"
#include <iostream>

Dense::Dense(int in, int out)
    : W(Matrix::random(in, out)),
      b(Matrix::zeros(1, out)),
      dW(in, out),
      db(1, out)
{}

Matrix& Dense::weights() { return W; }
Matrix& Dense::bias() { return b; }
Matrix& Dense::gradW() { return dW; }
Matrix& Dense::gradB() { return db; }

Matrix Dense::forward(const Matrix& x) {
    input = x;
    Matrix y = x * W;

    for(int i=0;i<y.rows;i++)
        for(int j=0;j<y.cols;j++)
            y(i,j) += b(0,j);

    return y;
}

Matrix Dense::backward(const Matrix& grad) {

    if (grad.cols != W.cols)
        throw runtime_error("Dense backward mismatch");

    // gradients
    dW = input.transpose() * grad;

    db = Matrix(1, grad.cols);
    for (int j = 0; j < grad.cols; j++) {
        double sum = 0.0;
        for (int i = 0; i < grad.rows; i++)
            sum += grad(i, j);
        db(0, j) = sum;
    }

    // propagate to previous layer
    Matrix grad_input = grad * W.transpose();
    return grad_input;
}


void Dense::update(double lr) {
    if(W.rows != dW.rows || W.cols != dW.cols)
        throw std::runtime_error("W/dW mismatch");

    if(b.rows != db.rows || b.cols != db.cols)
        throw std::runtime_error("b/db mismatch");

    W = W - dW * lr;
    b = b - db * lr;
}

void Dense::save(std::ofstream& f) {
    f.write((char*)W.data.data(),
            W.data.size()*sizeof(double));

    f.write((char*)b.data.data(),
            b.data.size()*sizeof(double));
}

void Dense::load(std::ifstream& f) {
    f.read((char*)W.data.data(),
           W.data.size()*sizeof(double));

    f.read((char*)b.data.data(),
           b.data.size()*sizeof(double));
}