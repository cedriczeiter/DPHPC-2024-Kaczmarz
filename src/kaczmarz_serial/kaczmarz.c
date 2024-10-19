#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kaczmarz_solver(double *A, double *b, double *x, unsigned rows, unsigned cols, unsigned max_iterations, double precision) {
  //printf("Starting Kaczmarz Solver...\n");
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    unsigned converged = 1;
    for (unsigned i = 0; i < rows; i++) {
      double dot_product = 0.0;
      double a_norm = 0.0;
      for (unsigned j = 0; j < cols; j++) {
        dot_product += A[i*cols + j] * x[j];
        a_norm += A[i*cols + j] * A[i*cols + j];
      }
      if (a_norm < 1e-10) {
        printf("Matrix row with 0 norm, iteration not possible.\n");
        return;
      }
      double correction = (b[i] - dot_product) / (a_norm);
      for (unsigned j = 0; j < cols; j++) {
        x[j] += A[i*cols + j] * correction;
      }
      if (fabs(correction) > precision) {
        converged = 0; //signal, that algorithm hasnt converged yet
      }
    }
    if (converged) return; //exit loop if fully converged
  }
  printf("Algorithm did not converge in %d iterations.\n", max_iterations);
}

