#include "matrix.h"
#include <cstdlib>
#include <stdexcept>
#include <iostream>

using namespace std;

Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r*c, 0.0) {}

double& Matrix::operator()(int r, int c) {
    if(r < 0 || r >= rows || c < 0 || c >= cols)
        throw runtime_error("Matrix index out of bounds");
    return data[r*cols + c];
}

const double& Matrix::operator()(int r, int c) const {
    if(r < 0 || r >= rows || c < 0 || c >= cols)
        throw runtime_error("Matrix index out of bounds");
    return data[r*cols + c];
}

Matrix Matrix::operator+(const Matrix& o) const {
    if(rows != o.rows || cols != o.cols)
        throw runtime_error("Matrix size mismatch in +");

    Matrix r(rows, cols);
    for(int i=0;i<rows*cols;i++)
        r.data[i] = data[i] + o.data[i];
    return r;
}

Matrix Matrix::operator-(const Matrix& o) const {
    if(rows != o.rows || cols != o.cols)
        throw std::runtime_error("Matrix size mismatch in -");

    Matrix r(rows, cols);
    for(int i=0;i<rows*cols;i++)
        r.data[i] = data[i] - o.data[i];
    return r;
}

Matrix Matrix::operator*(const Matrix& o) const {

    if(cols != o.rows) {
        std::cout << "MULTIPLY ERROR: "
                  << rows << "x" << cols
                  << " * "
                  << o.rows << "x" << o.cols
                  << std::endl;

        throw std::runtime_error("Matrix size mismatch in *");
    }

    Matrix r(rows, o.cols);

    #pragma omp parallel for
    for(int i=0; i<rows; i++) {
        int r_row_idx = i * o.cols;
        int t_row_idx = i * cols;
        for(int k=0; k<cols; k++) {
            double temp = data[t_row_idx + k];
            int o_row_idx = k * o.cols;
            
            // #pragma omp simd
            for(int j=0; j<o.cols; j++) {
                r.data[r_row_idx + j] += temp * o.data[o_row_idx + j];
            }
        }
    }

    return r;
}

Matrix Matrix::operator*(double s) const {
    Matrix r(rows, cols);
    for(int i=0;i<rows*cols;i++)
        r.data[i] = data[i] * s;
    return r;
}

Matrix Matrix::transpose() const {
    Matrix t(cols, rows);
    for(int i=0;i<rows;i++)
        for(int j=0;j<cols;j++)
            t(j,i) = (*this)(i,j);
    return t;
}

Matrix Matrix::random(int r, int c) {
    Matrix m(r,c);
    for(auto& x : m.data)
        x = ((double)rand()/RAND_MAX - 0.5);
    return m;
}

Matrix Matrix::zeros(int r, int c) {
    return Matrix(r,c);
}