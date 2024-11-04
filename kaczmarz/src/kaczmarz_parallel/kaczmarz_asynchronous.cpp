#include "kaczmarz_asynchronous.hpp"

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
                                              const double num_threads) {
  omp_set_num_threads(num_threads);

  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  std::uniform_int_distribution<> distr(0, rows - 1);

  //https://arxiv.org/pdf/2305.05482
  const unsigned L = 10000; //we check for convergence every 1000 steps
  double gamma = 1; //our initial step size; this step size will be updated during the solver run
  //note: the paper on asynchronous parallel kaczmarz proposes choosing a stepsize of gamma according to the maximal eigenvalue. Since we dont know the eigenvalues of our matrix without performing further expensive calculations, we can use adpative stepsize.
  //We adapt gamme according to this idea: 
  //if the residual goes down, we can maybe increase gamma a little: gamma *= 1.1
  //if the residual goes up, our stepsize is too large: gamme *= 0.5
  double current_residual = -1;
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

    unsigned local_rows;
    #pragma omp atomic read
    local_rows = rows;
    std::uniform_int_distribution<> distr(0, local_rows-1);

    auto b = lse.b();
    auto A = lse.A();

    for (unsigned iter = 0; iter < max_iterations / num_threads; iter++) {
      // Randomly select a row
      unsigned k = distr(rng);

#pragma omp atomic update
      total_iterations++;


      double dot_product = 0.0;
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it) {
        double x_value;
        omp_set_lock(&locks_x[it.col()]);
        x_value = x[it.col()];
        omp_unset_lock(&locks_x[it.col()]);
        dot_product += it.value() * x_value;
      }
      const double update_coeff = (b[k] - dot_product) / sq_norms[k];
      double stepsize;
      #pragma omp atomic read
      stepsize = gamma;
      //update
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
        /*//adapt stepsize
        if (residual < current_residual && current_residual >= 0){
          #pragma omp atomic update
          gamma *= 1.1;
        }
        else if (residual > current_residual && current_residual >= 0){
          #pragma omp atomic update
          gamma *= 0.5;
        }
        current_residual = residual;*/
        //std::cout << gamma << "                " << residual << std::endl;
      }
    }
  }

  if (converged) {
    std::cout << "\n\nDONE\n\n\n\n\n" << std::endl;
    return KaczmarzSolverStatus::Converged;
  }

  return KaczmarzSolverStatus::OutOfIterations;
}
