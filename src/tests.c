#include <stdio.h>
#include <math.h>
#include "linear_system.h"
#include "kaczmarz.h"
#include "tests.h"

void run_tests() {
    printf("Running tests...\n");

    // Test 1: Small system with known solution
    int rows = 3;
    int cols = 3;
    LinearSystem *sys = allocate_system(rows, cols);

    // A = [[3, 2, -1], [2, -2, 4], [-1, 0.5, -1]]
    double A[3][3] = {
        {3, 2, -1},
        {2, -2, 4},
        {-1, 0.5, -1}
    };
    double b[3] = {1, -2, 0};
    double expected_solution[3] = {1, -2, -2};

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sys->A[i][j] = A[i][j];
        }
        sys->b[i] = b[i];
        sys->x[i] = 0.0;
    }

    // Call parallelized Kaczmarz solver
    kaczmarz_solver(sys);

    // Check results
    int passed = 1;
    for (int i = 0; i < rows; i++) {
        if (fabs(sys->x[i] - expected_solution[i]) > 1e-3) {
            passed = 0;
        }
        printf("x[%d] = %f\n", i, sys->x[i]);
    }

    if (passed) {
        printf("Test 1 passed.\n");
    } else {
        printf("Test 1 failed.\n");
    }

    // Deallocate system
    deallocate_system(sys);

    printf("Tests completed.\n");
}