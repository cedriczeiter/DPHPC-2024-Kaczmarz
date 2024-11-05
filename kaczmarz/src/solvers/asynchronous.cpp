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

  const unsigned L = 1000;  // we check for convergence every 10000 steps
  unsigned total_iterations = 0;
  bool converged = false;
  double current_residual = 1000;  // arbitrarily set
  double stepsize_global = 1;

  const unsigned runs_per_thread = 30;

  // we create locks for each entry in x (so the
  // threads block each other as little as possible)
  std::vector<omp_lock_t> locks_x(cols);
  for (unsigned i = 0; i < cols; ++i) {
    omp_init_lock(&locks_x[i]);
  }

  Vector x_momentum = Eigen::VectorXd::Zero(cols);
  const double beta = 0;

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    if (sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }

  // each thread chooses randomly from own set of rows
  std::vector<std::vector<unsigned> > local_rows(num_threads);
  for (int j = 0; j < num_threads; j++) {
    for (unsigned i = j; i < rows;
         i += num_threads) {
      local_rows[j].push_back(i);
    }
  }

    auto A = lse.A();
      auto b = lse.b();
      std::mt19937 rng(21);

  for (unsigned iter = 0; iter < max_iterations; iter++) {
#pragma omp parallel
    {

      const unsigned thread_num = omp_get_thread_num();
      const unsigned local_rows_size = local_rows[thread_num].size();
      std::uniform_int_distribution<> distr(0, local_rows_size - 1);

      for (int i = 0; i < runs_per_thread; i++) {
        // Randomly select a row
        unsigned k = local_rows[thread_num].at(distr(rng));

        double dot_product = 0.0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
                 A, k);
             it; ++it) {
          double x_value;
          omp_set_lock(&locks_x[it.col()]);
          x_value = x[it.col()];
          omp_unset_lock(&locks_x[it.col()]);
          dot_product += it.value() * x_value;
        }


        const double update_coeff = ((b[k] - dot_product) / sq_norms[k]);
        // update
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
          omp_set_lock(&locks_x[it.col()]);
          const double update =
              update_coeff * it.value() + beta * x_momentum[it.col()];
          x[it.col()] += update;
          x_momentum[it.col()] = update;
          omp_unset_lock(&locks_x[it.col()]);
        }

      }
    }

    // stopping criterion
    if (iter % L == 0 && iter > 0) {  // Check every L iterations
      double residual = (lse.A() * x - lse.b()).norm();
      if (residual < precision) {
        return KaczmarzSolverStatus::Converged;
      }
      //std::cout << residual << "        " << std::endl;
    }
  }

  return KaczmarzSolverStatus::OutOfIterations;
}
