#include "asynchronous.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "common.hpp"

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads) {
  assert(omp_get_cancellation());  // we need cancellation enabled for this
                                   // algorithm
  omp_set_num_threads(num_threads);

  const unsigned rows = lse.row_count();
  // const unsigned cols = lse.column_count();

  assert(num_threads <=
         rows);  // necessary for allowing each thread to have local rows

  const unsigned L = 500;  // we check for convergence every L steps
  bool converged = false;

  const unsigned runs_per_thread = 15;

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    if (sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }

  // each thread chooses randomly from own set of rows
  unsigned rows_per_thread = (unsigned)(rows / num_threads);
  std::vector<std::vector<unsigned> > local_rows(num_threads);
  for (unsigned i = 0; i < num_threads; i++) {
    for (unsigned j = rows_per_thread * i;
         j < rows && j < rows_per_thread * (i + 1); j++) {
      local_rows.at(i).push_back(j);
    }
  }
  for (unsigned j = rows_per_thread * num_threads; j < rows; j++) {
    local_rows.at(num_threads - 1).push_back(j);
  }

#pragma omp parallel
  {
    const auto A = lse.A();
    const auto b = lse.b();
    std::mt19937 rng(21);

    const unsigned thread_num = omp_get_thread_num();
    const unsigned local_rows_size = local_rows[thread_num].size();
    std::uniform_int_distribution<> distr(0, local_rows_size - 1);

    // Loop performed by all threads
    for (unsigned iter = 0; iter < max_iterations; iter++) {
      for (unsigned i = 0; i < runs_per_thread; i++) {
        // Randomly select a row
        unsigned k = local_rows[thread_num].at(distr(rng));

        double dot_product = 0.0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A,
                                                                            k);
             it; ++it) {
          double x_value;
#pragma omp atomic read
          x_value = x[it.col()];
          dot_product += it.value() * x_value;
        }

        double update_coeff = ((b[k] - dot_product) / sq_norms[k]);
        //  update
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
          const double update = update_coeff * it.value();
#pragma omp atomic update
          x[it.col()] += update;
        }
      }
// we synchronize all threads, so we comply with the convergence
// conditions of the asynchronous algorithms
#pragma omp barrier

      // stopping criterion
      if (thread_num == 0 && iter % L == 0 &&
          iter > 0) {  // Check every L iterations
        double residual = (A * x - b).norm();
        if (residual < precision) {
#pragma omp atomic write
          converged = true;
        }
        // std::cout << residual << std::endl;
      }
      if (converged) {
#pragma omp cancel parallel
      }

#pragma omp cancellation point parallel
    }
  }
  if (converged) return KaczmarzSolverStatus::Converged;

  return KaczmarzSolverStatus::OutOfIterations;
}
