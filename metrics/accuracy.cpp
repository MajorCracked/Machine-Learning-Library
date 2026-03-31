#include "accuracy.h"
#include <cmath>

double Accuracy::compute(const Matrix& pred,
                         const Matrix& target) {

    int correct = 0;

    for(int i=0;i<pred.rows;i++) {

        int p = pred(i,0) > 0.5 ? 1 : 0;
        int t = target(i,0);

        if(p == t)
            correct++;
    }

    return (double)correct / pred.rows;
}