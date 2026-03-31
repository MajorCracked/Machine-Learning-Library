#pragma once
#include <vector>
using namespace std;

class Matrix {
public:
    int rows, cols;
    vector<double> data;


    Matrix(int r=0, int c=0);

    double& operator()(int r, int c);
    const double& operator()(int r, int c) const;

    Matrix operator+(const Matrix& o) const;
    Matrix operator-(const Matrix& o) const;
    Matrix operator*(const Matrix& o) const;

    Matrix operator*(double s) const;

    Matrix transpose() const;

    static Matrix random(int r, int c);
    static Matrix zeros(int r, int c);
};