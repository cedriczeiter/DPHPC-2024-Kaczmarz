#include <stdio.h>
#include "linear_system.h"
#include "kaczmarz.h"
#include "pde_solver.h"

#define N 1000

void simple_pde_solver() {
    // A basic representation of Ax = b for PDE solution
    int rows = N;
    int cols = N;
    LinearSystem *sys = allocate_system(rows, cols);

    // Initialize A, b, x (here using arbitrary values for demonstration)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sys->A[i][j] = (i == j) ? 2.0 : 1.0; // A simple symmetric matrix
        }
        sys->b[i] = (double)(i + 1);
        sys->x[i] = 0.0;
    }

    // Call Kaczmarz solver
    kaczmarz_solver(sys);

    // Display results (first 10 values of x)
    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %f\n", i, sys->x[i]);
    }

    // Deallocate system
    deallocate_system(sys);
}