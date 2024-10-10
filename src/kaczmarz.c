#include <math.h>
#include "kaczmarz.h"

#define MAX_ITER 1000

void kaczmarz_solver(LinearSystem *sys) {
    for (int iter = 0; iter < MAX_ITER; iter++) {
        for (int i = 0; i < sys->rows; i++) {
            double dot_product = 0.0;
            for (int j = 0; j < sys->cols; j++) {
                dot_product += sys->A[i][j] * sys->x[j];
            }
            double correction = (sys->b[i] - dot_product) / (dot_product + 1e-6);
            for (int j = 0; j < sys->cols; j++) {
                sys->x[j] += sys->A[i][j] * correction;
            }
        }
    }
}