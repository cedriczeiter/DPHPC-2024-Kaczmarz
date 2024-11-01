#include "kaczmarz_asynchronous.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <random>

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              double *x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const double num_threads) {
  omp_set_num_threads(num_threads);

  std::mt19937 rng(1);
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  std::uniform_int_distribution<> distr(0, rows - 1);

  // vector for storing convergence criterions
  std::vector<double> prev_x(cols, 0.0);
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
    for (unsigned iter = 0; iter < max_iterations / num_threads; iter++) {
      bool substantial_correction = false;

      // Randomly select a row based on the squared norms
      unsigned k = distr(rng);

      // Access the selected row
      const auto a_row = lse.A().row(k);

      // Compute the dot product and row squared norm
      double dot_product = 0.0;
      double a_norm = sq_norms[k];

      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
               lse.A(), k);
           it; ++it) {
        double x_value;
#pragma omp atomic read
        x_value = x[it.col()];
        dot_product += it.value() * x_value;
      }
      const double update_coeff = (lse.b()[k] - dot_product) / sq_norms[k];
      for (SparseMatrix::InnerIterator it(lse.A(), k); it; ++it) {
#pragma omp atomic update
        x[it.col()] += update_coeff * it.value();
      }

      // Stop if a row squared norm of a row is zero
      if (a_norm < 1e-10) {
        // return KaczmarzSolverStatus::ZeroNormRow; //TODO: deal with this
      }

      // check if stopping criterion has been reached
      bool local_converged;
#pragma omp atomic read
      local_converged = converged;
      if (local_converged) break;

      // thread 0 applies LISE stopping criterion
      if (omp_get_thread_num() == 0 && iter % L == 0 &&
          iter > 0) {  // Check every L iterations
        double norm_diff = 0.0;
        for (unsigned j = 0; j < cols; j++) {
          double x_value;
#pragma omp atomic read
          x_value = x[j];
          double diff = x_value - prev_x[j];
          norm_diff += diff * diff;
        }
        norm_diff = std::sqrt(norm_diff);
        // Check if the LISE stopping criterion is met
        if (norm_diff < precision) {
          converged = true;
        }
        // Update prev_x to store the current solution
        for (unsigned j = 0; j < cols; j++) {
          double x_value;
#pragma omp atomic read
          x_value = x[j];
          prev_x[j] = x_value;
        }
      }
    }
  }

  if (converged) return KaczmarzSolverStatus::Converged;

  return KaczmarzSolverStatus::OutOfIterations;
}
