#include "kaczmarz_asynchronous.hpp"

#include <omp.h>

#include <cmath>
#include <cstdlib>
#include <random>

KaczmarzSolverStatus sparse_kaczmarz(const SparseLinearSystem &lse, double *x,
                                     const unsigned max_iterations,
                                     const double precision,
                                     const double num_threads) {
  std::mt19937 rng(1);
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  std::uniform_int_distribution<> distr(0, rows - 1);

  // init lock used for writing to x
  omp_lock_t writelock;
  omp_init_lock(&writelock);

  // vector for storing convergence criterions
  std::vector prev_x(cols);
  std::copy(x, x + cols, prev_x.begin());
  // L for LISE convergence criterion
  const unsigned L = 50;
  bool converged = false;

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  }

#pragma omp parallel
  {
    for (unsigned iter = 0; iter < max_iterations / num_threads; ier++) {
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

      const double correction = (lse.b()[i] - dot_product) / a_norm;

#pragma omp critical
      {
        for (unsigned j = 0; j < cols; j++) {
          x[j] += a_row[j] * correction;
        }
      }

      // check if stopping criterion has been reached
      if (converged) break;

      // thread 0 applies LISE stopping criterion
      if (omp_get_thread_num() == 0 && iter % L == 0 &&
          iter > 0) {  // Check every L iterations
        double norm_diff = 0.0;
        for (unsigned j = 0; j < cols; j++) {
          double diff = x[j] - prev_x[j];
          norm_diff += diff * diff;
        }
        norm_diff = std::sqrt(norm_diff) / L;

        // Check if the LISE stopping criterion is met
        if (norm_diff < precision) {
          converged = true;
        }

        // Update prev_x to store the current solution
        std::copy(x, x + cols, prev_x.begin());
      }
    }
  }

  if (converged) return KaczmarzSolverStatus::Converged;

  return KaczmarzSolverStatus::OutOfIterations;
}
