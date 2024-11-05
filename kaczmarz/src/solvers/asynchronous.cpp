#include "asynchronous.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads) {
  omp_set_num_threads(num_threads);

  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();

  const unsigned L = 10000;  // we check for convergence every 10000 steps
  unsigned total_iterations = 0;
  bool converged = false;

  // we create locks for each entry in x and each entry in prev_x (so the
  // threads block each other as little as possible)
  std::vector<omp_lock_t> locks_x(cols);
  for (unsigned i = 0; i < cols; ++i) {
    omp_init_lock(&locks_x[i]);
  }

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    if (sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }

#pragma omp parallel
  {
    unsigned thread_num = omp_get_thread_num();
    std::mt19937 rng(thread_num);

    // each thread chooses randomly from own set of rows
    std::vector<unsigned> local_rows;
    for (unsigned i = thread_num; i < rows; i += omp_get_num_threads()) {
      local_rows.push_back(i);
    }

    const unsigned local_rows_size = local_rows.size();
    std::uniform_int_distribution<> distr(0, local_rows_size - 1);

    auto b = lse.b();
    auto A = lse.A();

    for (unsigned iter = 0; iter < max_iterations; iter++) {
      // Randomly select a row
      unsigned k = local_rows.at(distr(rng));

#pragma omp atomic update
      total_iterations++;

      double dot_product = 0.0;
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k);
           it; ++it) {
        double x_value;
        omp_set_lock(&locks_x[it.col()]);
        x_value = x[it.col()];
        omp_unset_lock(&locks_x[it.col()]);
        dot_product += it.value() * x_value;
      }
      const double update_coeff = (b[k] - dot_product) / sq_norms[k];
      // update
      for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
        double update = update_coeff * it.value();
        omp_set_lock(&locks_x[it.col()]);
        x[it.col()] += update;
        omp_unset_lock(&locks_x[it.col()]);
      }

      // check if stopping criterion has been reached
      bool local_converged;
#pragma omp atomic read
      local_converged = converged;
      if (local_converged) break;

      // thread 0 applies stopping criterion
      if (omp_get_thread_num() == 0 && total_iterations % L == 0 &&
          total_iterations > 0) {  // Check every L iterations
        for (unsigned i = 0; i < cols; ++i) {
          omp_set_lock(&locks_x[i]);
        }
        double residual = (A * x - b).norm();
        if (residual < precision) {
#pragma omp atomic write
          converged = true;
        }
        for (unsigned i = 0; i < cols; ++i) {
          omp_unset_lock(&locks_x[i]);
        }
      }
    }
  }

  if (converged) {
    return KaczmarzSolverStatus::Converged;
  }

  return KaczmarzSolverStatus::OutOfIterations;
}
