#include "basic_cuda.hpp"

double updateAllRowsParallel(const DenseLinearSystem &lse, double *x,
                             const unsigned rows, const unsigned cols) {
  double smallestRowSqNorm = 1e10;
  #pragma omp parallel for reduction(min:smallestRowSqNorm)
  for (unsigned i = 0; i < rows; i++) {
    const double *const a_row = lse.A() + i * cols;
    double dot_product = 0.0;
    double row_sq_norm = 0.0;

    // Compute the dot product and row squared norm
    for (unsigned j = 0; j < cols; j++) {
      dot_product += a_row[j] * x[j];
      row_sq_norm += a_row[j] * a_row[j];
    }

    // Stop if a row squared norm of a row is zero
    if (row_sq_norm < 1e-10) {
      smallestRowSqNorm = 0;
    }

    const double correction = (lse.b()[i] - dot_product) / row_sq_norm;
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }
  }
  return smallestRowSqNorm;
}