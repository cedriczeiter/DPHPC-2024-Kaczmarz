#include "random_kaczmarz.hpp"

#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>
#include <numeric>

// Helper function to randomly select a row based on row norms
unsigned random_row_selection(const double *row_norms, const unsigned num_rows,
                              std::mt19937 &rng) {
  std::discrete_distribution<> dist(row_norms, row_norms + num_rows);  // Distribution based on row norms
  return dist(rng);                      // Randomly select a row
}

KaczmarzSolverStatus kaczmarz_random_solver(const DenseLinearSystem &lse,
                                            double *x, unsigned max_iterations,
                                            double precision) {
  std::mt19937 rng(1);
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();

  // Precompute row norms (squared)
  std::vector<double> row_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    const double *const a_row = lse.A() + i * cols;
    double row_sq_norm = 0.0;
    for (unsigned j = 0; j < cols; j++) {
      row_sq_norm += a_row[j] * a_row[j];
    }
    row_norms[i] = row_sq_norm;
  }

  // Iterate through a maximum of max_iterations
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_correction = false;  // Track significant updates to x

    // Randomly select a row based on the squared norms
    unsigned i = random_row_selection(row_norms.data(), rows, rng);

    // Access the selected row
    const double *const a_row = lse.A() + i * cols;

    // Compute the dot product and row squared norm
    double dot_product = 0.0;
    double a_norm = row_norms[i];
    for (unsigned j = 0; j < cols; j++) {
      dot_product += a_row[j] * x[j];
    }

    // Stop if a row squared norm of a row is zero
    if (a_norm < 1e-10) {
      return KaczmarzSolverStatus::ZeroNormRow;
    }

    // Compute correction for the selected row
    const double correction = (lse.b()[i] - dot_product) / a_norm;
    if (std::fabs(correction) > precision) {
      substantial_correction = true;  // Mark substantial change
    }

    // Update the solution vector x with the correction
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }

    // Calculate the residual norm to check for convergence
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

    // If residual norm is less than the squared precision, declare convergence
    if (std::sqrt(residual_norm_sq) < precision) {
      return KaczmarzSolverStatus::Converged;
    }
  }

  // If max iterations reached without convergence, return OutOfIterations
  return KaczmarzSolverStatus::OutOfIterations;
}
