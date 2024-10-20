#include "kaczmarz.hpp"

#include <cmath>

KaczmarzSolverStatus dense_kaczmarz(const DenseLinearSystem& lse, double *x, unsigned max_iterations, double precision) {
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  // Iterate through a maximum of max_iterations
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    // the algorithm has converged iff none of the rows in an iteration caused a substantial correction
    bool substantial_correction = false;

    // Process each row of matrix A
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
        return KaczmarzSolverStatus::ZeroNormRow;
      }

      // Check if the correction is substantial
      const double correction = (lse.b()[i] - dot_product) / row_sq_norm;
      for (unsigned j = 0; j < cols; j++) {
        x[j] += a_row[j] * correction;
      }
      if (std::fabs(correction) > precision) {
        substantial_correction = true;
      }
    }

    // If no substantial correction was made, the solution has converged and algorithm ends
    if (!substantial_correction) {
      return KaczmarzSolverStatus::Converged;
    }
  }

  //If it didnt return earlier, then max iterations reached and not converged.
  return KaczmarzSolverStatus::OutOfIterations;
}

KaczmarzSolverStatus sparse_kaczmarz(const SparseLinearSystem& lse, Eigen::VectorXd& x, const unsigned max_iterations, const double precision) {
  const unsigned rows = lse.row_count();
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  }
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_update = false;
    for (unsigned i = 0; i < rows; i++) {
      const auto row = lse.A().row(i);
      const double update_coeff = (lse.b()[i] - row.dot(x)) / sq_norms[i];
      x += update_coeff * row;
      if (std::fabs(update_coeff) > precision) {
        substantial_update = true;
      }
    }
    if (!substantial_update) {
      return KaczmarzSolverStatus::Converged;
    }
  }
  return KaczmarzSolverStatus::OutOfIterations;
}
