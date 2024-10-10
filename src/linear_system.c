#include <stdlib.h>
#include "linear_system.h"

LinearSystem* allocate_system(int rows, int cols) {
    LinearSystem *sys = (LinearSystem*) malloc(sizeof(LinearSystem));
    sys->rows = rows;
    sys->cols = cols;

    sys->A = (double**) malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        sys->A[i] = (double*) malloc(cols * sizeof(double));
    }

    sys->b = (double*) malloc(rows * sizeof(double));
    sys->x = (double*) malloc(cols * sizeof(double));
    return sys;
}

void deallocate_system(LinearSystem *sys) {
    for (int i = 0; i < sys->rows; i++) {
        free(sys->A[i]);
    }
    free(sys->A);
    free(sys->b);
    free(sys->x);
    free(sys);
}