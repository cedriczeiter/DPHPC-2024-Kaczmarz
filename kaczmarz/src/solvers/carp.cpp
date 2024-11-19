#include "carp.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "carp_cuda.hpp"
#include "common.hpp"
#include "basic.hpp"

KaczmarzSolverStatus carp_gpu(const SparseLinearSystem& lse, Vector& x,
                              const unsigned max_iterations,
                              const double precision,
                              const unsigned num_threads) {
  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration

  const int* A_outer = lse.A().outerIndexPtr();
  const int* A_inner = lse.A().innerIndexPtr();
  const double* A_values = lse.A().valuePtr();

  const double* b = lse.b().data();
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  const unsigned nnz = lse.A().nonZeros();

  std::vector<double> h_sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    h_sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    if (h_sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }

  //get maximum nr of nnz in row
  int max_nnz_in_row = 0;
  for (unsigned i = 0; i < rows; i++){
    //std::cout << "index: " << i << std::endl;
    const int nnz_in_row = A_outer[i+1] - A_outer[i];
    if (nnz_in_row > max_nnz_in_row) max_nnz_in_row = nnz_in_row;
  }
  //std::cout << "MAX NNZ: " << max_nnz_in_row << std::endl;


  //call carp solver for beginning
  return invoke_carp_solver_gpu(A_outer, A_inner, A_values, b, x.data(),
                                        h_sq_norms.data(), rows, cols, nnz,
                                        max_iterations, precision, max_nnz_in_row);
}