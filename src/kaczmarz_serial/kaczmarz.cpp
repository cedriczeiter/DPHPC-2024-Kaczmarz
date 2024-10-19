#include "kaczmarz.hpp"

#include <cmath>

KaczmarzSolverStatus kaczmarz_solver(const double *A, const double *b, double *x, unsigned rows, unsigned cols, unsigned max_iterations, double precision) {
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    // the algorithm has converged iff none of the rows in an iteration caused a substantial correction
    bool substantial_correction = false;
    for (unsigned i = 0; i < rows; i++) {
      double dot_product = 0.0;
      double row_sq_norm = 0.0;
      const double *const a_row = A + i * cols;
      for (unsigned j = 0; j < cols; j++) {
        dot_product += a_row[j] * x[j];
        row_sq_norm += a_row[j] * a_row[j];
      }
      if (row_sq_norm < 1e-10) {
        return KaczmarzSolverStatus::ZeroNormRow;
      }
      const double correction = (b[i] - dot_product) / row_sq_norm;
      for (unsigned j = 0; j < cols; j++) {
        x[j] += A[i*cols + j] * correction;
      }
      if (std::fabs(correction) > precision) {
        substantial_correction = true;
      }
    }
    if (!substantial_correction) {
      return KaczmarzSolverStatus::Converged;
    }
  }
  return KaczmarzSolverStatus::OutOfIterations;
}

