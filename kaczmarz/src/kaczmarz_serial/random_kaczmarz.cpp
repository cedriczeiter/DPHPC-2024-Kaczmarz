#include "random_kaczmarz.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <random>
#include <vector>

// Helper function to randomly select a row based on row norms
unsigned random_row_selection(const double *row_norms, const unsigned num_rows,
                              std::mt19937 &rng) {
  std::discrete_distribution<> dist(
      row_norms, row_norms + num_rows);  // Distribution based on row norms
  return dist(rng);                      // Randomly select a row
}

// Main function with the LISE stopping criterion
KaczmarzSolverStatus kaczmarz_random_solver(
    const DenseLinearSystem &lse, double *x, unsigned max_iterations,
    double precision, std::vector<double> &times_residuals,
    std::vector<double> &residuals, std::vector<int> &iterations,
    const int convergence_step_rate) {
  std::mt19937 rng(41);
  unsigned L = 50;
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
  std::vector<double> prev_x(cols, 0.0);  // Store solution every L iterations

  // Iterate through a maximum of max_iterations
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_correction = false;

    if (iter % convergence_step_rate == 0) {
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      times_residuals.push_back(elapsed.count());

      // Calculate the residual norm
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
      // if (residual_fraction <= precision) {
      //   return KaczmarzSolverStatus::Converged;
      // }
      if (elapsed.count() >= 100.595) {
        return KaczmarzSolverStatus::OutOfIterations;
      }
    }

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

    // Apply Kaczmarz correction
    const double correction = (lse.b()[i] - dot_product) / a_norm;
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }

    // LISE Stopping Criterion
    if (iter % L == 0 && iter > 0) {  // Check every L iterations
      double norm_diff = 0.0;
      for (unsigned j = 0; j < cols; j++) {
        double diff = x[j] - prev_x[j];
        norm_diff += diff * diff;
      }
      norm_diff = std::sqrt(norm_diff) / L;

      // Check if the LISE stopping criterion is met
      if (norm_diff < precision) {
        return KaczmarzSolverStatus::Converged;
      }

      // Update prev_x to store the current solution
      std::copy(x, x + cols, prev_x.begin());
    }
  }

  // If it didn't return earlier, then max iterations were reached without
  // convergence
  return KaczmarzSolverStatus::OutOfIterations;
}
