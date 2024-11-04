#include "basic.hpp"

#include <chrono>
#include <cmath>

KaczmarzSolverStatus dense_kaczmarz(const DenseLinearSystem &lse, double *x,
                                    const unsigned max_iterations,
                                    const double precision,
                                    std::vector<double> &times_residuals,
                                    std::vector<double> &residuals,
                                    std::vector<int> &iterations,
                                    const int convergence_step_rate) {
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();

  // Calculate the residual norm in the beginning to check for convergence
  double residual_norm_sq = 0.0;
  for (unsigned k = 0; k < rows; k++) {
    double row_residual = 0.0;
    const double *row = lse.A() + k * cols;
    for (unsigned j = 0; j < cols; j++) {
      row_residual += row[j] * x[j];
    }
    row_residual -= lse.b()[k];
    residual_norm_sq += row_residual * row_residual;
  }

  const double residual_norm_0 = std::sqrt(residual_norm_sq);
  const auto start = std::chrono::high_resolution_clock::now();

  // Iterate through a maximum of max_iterations
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    // the algorithm has converged iff none of the rows in an iteration caused a
    // substantial correction
    bool substantial_correction = false;

    if (iter % convergence_step_rate == 0) {
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      times_residuals.push_back(elapsed.count());
      double residual_norm_sq = 0.0;
      for (unsigned k = 0; k < rows; k++) {
        double row_residual = 0.0;
        const double *row = lse.A() + k * cols;
        for (unsigned j = 0; j < cols; j++) {
          row_residual += row[j] * x[j];
        }
        row_residual -= lse.b()[k];
        residual_norm_sq += row_residual * row_residual;
      }

      double residual_norm_now = std::sqrt(residual_norm_sq);
      double residual_fraction = residual_norm_now / residual_norm_0;
      residuals.push_back(residual_fraction);
      iterations.push_back(iter);
    }

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

    // If no substantial correction was made, the solution has converged and
    // algorithm ends
    if (!substantial_correction) {
      return KaczmarzSolverStatus::Converged;
    }
  }

  // If it didnt return earlier, then max iterations reached and not converged.
  return KaczmarzSolverStatus::OutOfIterations;
}

KaczmarzSolverStatus sparse_kaczmarz(
    const SparseLinearSystem &lse, Eigen::VectorXd &x,
    const unsigned max_iterations, const double precision,
    std::vector<double> &times_residuals, std::vector<double> &residuals,
    std::vector<int> &iterations, const int convergence_step_rate) {
  const unsigned rows = lse.row_count();
  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  }

  // Calculate the initial residual norm to check for convergence
  double residual_norm_sq = 0.0;

  // // Compute squared norms of rows of A
  // for (unsigned i = 0; i < rows; i++) {
  //     sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  // }

  for (unsigned k = 0; k < rows; k++) {
    double row_residual = 0.0;
    // Access non-zero elements by iterating through the outer index
    for (Eigen::Index j = lse.A().outerIndexPtr()[k];
         j < lse.A().outerIndexPtr()[k + 1]; ++j) {
      row_residual += lse.A().valuePtr()[j] * x[lse.A().innerIndexPtr()[j]];
    }
    row_residual -= lse.b()[k];
    residual_norm_sq += row_residual * row_residual;
  }

  const double residual_norm_0 = std::sqrt(residual_norm_sq);
  const auto start = std::chrono::high_resolution_clock::now();

  // same algorithm as in the dense case
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_update = false;

    if (iter % convergence_step_rate == 0) {
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      times_residuals.push_back(elapsed.count());
      // Calculate the initial residual norm to check for convergence
      double residual_norm_sq = 0.0;
      // Vector sq_norms(rows);

      // // Compute squared norms of rows of A
      // for (unsigned i = 0; i < rows; i++) {
      //     sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
      // }

      for (unsigned k = 0; k < rows; k++) {
        double row_residual = 0.0;

        // Access non-zero elements by iterating through the outer index
        for (Eigen::Index j = lse.A().outerIndexPtr()[k];
             j < lse.A().outerIndexPtr()[k + 1]; ++j) {
          row_residual += lse.A().valuePtr()[j] * x[lse.A().innerIndexPtr()[j]];
        }
        row_residual -= lse.b()[k];
        residual_norm_sq += row_residual * row_residual;
      }

      double residual_norm_now = std::sqrt(residual_norm_sq);
      double residual_fraction = residual_norm_now / residual_norm_0;
      residuals.push_back(residual_fraction);
      iterations.push_back(iter);
    }
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
