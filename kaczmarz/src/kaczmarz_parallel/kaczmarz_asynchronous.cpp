#include "kaczmarz_asynchronous.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              double *x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const double num_threads) {
  omp_set_num_threads(num_threads);


  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  std::uniform_int_distribution<> distr(0, rows - 1);


  // L for LISE convergence criterion
  const unsigned L = 50;
  bool converged = false;
  std::vector<double> updates(cols, 0.0);

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  Vector sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  }

#pragma omp parallel
  {
    std::mt19937 rng(omp_get_thread_num());
    auto b = lse.b();
    auto A = lse.A();

    for (unsigned iter = 0; iter < max_iterations / num_threads; iter++) {

      // Randomly select a row based on the squared norms
      unsigned k = distr(rng);

      // Compute the dot product and row squared norm
      double dot_product = 0.0;
      double a_norm = sq_norms[k];


      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(
               A, k);
           it; ++it) {

        double x_value;
#pragma omp atomic read
        x_value = x[it.col()];

        dot_product += it.value() * x_value;
      }

      const double update_coeff = (b[k] - dot_product) / sq_norms[k];
      for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
        double update = update_coeff * it.value();

#pragma omp atomic update
        x[it.col()] += update;
#pragma omp atomic update
        updates[i] += update;
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
        double tot_updates;
        #pragma omp atomic read
        tot_updates = updates;

        
        tot_updates = tot_updates / L;
        
        if (std::sqrt(tot_updates) < precision){
          #pragma omp atomic write
          converged = true;
          std::cout << "Total updates:" << std::sqrt(tot_updates) << " Precision: " << precision << " CONVERGED" << std::endl;
        } 
        std::cout << "Total updates:" << std::sqrt(tot_updates) << " Precision: " << precision << " WORKING" << std::endl;
        #pragma omp atomic write
        updates = 0;
      }
    }
  }

  if (converged) return KaczmarzSolverStatus::Converged;

  return KaczmarzSolverStatus::OutOfIterations;
}
