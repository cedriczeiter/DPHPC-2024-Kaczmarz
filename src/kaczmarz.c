#include "kaczmarz.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void kaczmarz_solver(LinearSystem *sys, int max_iterations, double precision) {
  for (int iter = 0; iter < max_iterations; iter++) {
    for (int i = 0; i < sys->rows; i++) {
      double dot_product = 0.0;
      double a_norm = 0.0;
      for (int j = 0; j < sys->cols; j++) {
        dot_product += sys->A[i][j] * sys->x[j];
        a_norm += sys->A[i][j] * sys->A[i][j];
      }
      if (a_norm < 1e-10) {
        printf("Matrix column with 0 norm, iteration not possible.");
        return;
      }
      double correction = (sys->b[i] - dot_product) / (a_norm);
      for (int j = 0; j < sys->cols; j++) {
        sys->x[j] += sys->A[i][j] * correction;
      }
      if (fabs(correction) < precision) {
        return;
      }
    }
  }
  printf("Algorithm did not converge in %d iterations.", max_iterations);
}

// helper function: select the row in the random scheme
int random_row_selection(double *row_norms, int num_rows) {
  // Calculation of the norm of the rows
  double total_norm = 0.0;
  for (int i = 0; i < num_rows; i++) {
    total_norm += row_norms[i];
  }

  // r chosen proportional to ||a_{r(i)}||_2^2 (see paper)
  double r = ((double)rand() / RAND_MAX) * total_norm;
  double cumulative_norm = 0.0;

  // iterate through it based on the random number
  for (int i = 0; i < num_rows; i++) {
    cumulative_norm += row_norms[i];
    if (r <= cumulative_norm) {
      return i;
    }
  }
  return num_rows - 1; // in case of an error
}

void kaczmarz_random_solver(LinearSystem *sys, int max_iterations, double precision) {
  // seeding! 
  srand(1);

  // precompute row norms (squared) ->> can be parallelized ?
  double *row_norms = (double *)malloc(sys->rows * sizeof(double));
  for (int i = 0; i < sys->rows; i++) {
    double norm = 0.0;
    for (int j = 0; j < sys->cols; j++) {
      norm += sys->A[i][j] * sys->A[i][j];
    }
    row_norms[i] = norm;
  }

  // perform iterations
  for (int iter = 0; iter < max_iterations; iter++) {
    // call helper function to know which one to call
    int i = random_row_selection(row_norms, sys->rows);

    // regular method
    double dot_product = 0.0;
    double a_norm = row_norms[i];
    for (int j = 0; j < sys->cols; j++) {
      dot_product += sys->A[i][j] * sys->x[j];
    }
    if (a_norm < 1e-10) {
      printf("Matrix column with 0 norm, iteration not possible.");
      free(row_norms);
      return;
    }
    double correction = (sys->b[i] - dot_product) / (a_norm);
    for (int j = 0; j < sys->cols; j++) {
      sys->x[j] += sys->A[i][j] * correction;
    }
    if (fabs(correction) < precision) {
      free(row_norms);
      return;
    }
  }

  printf("Algorithm did not converge in %d iterations.", max_iterations);
  free(row_norms);
}
