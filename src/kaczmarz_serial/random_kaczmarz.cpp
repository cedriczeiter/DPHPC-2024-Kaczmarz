#include "kaczmarz.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>

// Helper function to randomly select a row based on row norms
unsigned random_row_selection(const double *row_norms, unsigned num_rows, std::mt19937 &rng) {
  std::discrete_distribution<> dist(row_norms, row_norms + num_rows); // Distribution based on row norms
  return dist(rng);  // Randomly select a row
}

KaczmarzSolverStatus kaczmarz_random_solver(const DenseLinearSystem& lse, double *x, unsigned max_iterations, double precision) {
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
    bool substantial_correction = false;

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

    // Check if the correction is substantial
    const double correction = (lse.b()[i] - dot_product) / a_norm;
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }
    if (std::fabs(correction) > precision) {
      substantial_correction = true;
    }

    // If no substantial correction was made, the solution has converged and algorithm ends
    if (!substantial_correction) {
      return KaczmarzSolverStatus::Converged;
    }
  }

  //If it didnt return earlier, then max iterations reached and not converged.
  return KaczmarzSolverStatus::OutOfIterations;
}
