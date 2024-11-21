#include "carp.hpp"

#include <omp.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "basic.hpp"
#include "carp_cuda.hpp"
#include "common.hpp"

KaczmarzSolverStatus carp_gpu(const SparseLinearSystem& lse, Vector& x,
                              const unsigned max_iterations,
                              const double precision) {
  // get the sparse matrix in CSR format
  const int* A_outer =
      lse.A().outerIndexPtr();  // outer index of the sparse matrix in CSR
                                // format (row pointer)
  const int* A_inner =
      lse.A().innerIndexPtr();  // inner index of the sparse matrix in CSR
                                // format (column indices)
  const double* A_values =
      lse.A().valuePtr();  // non-zero values of the sparse matrix in CSR format

  // get the right-hand side vector
  const double* b = lse.b().data();

  const double b_norm = lse.b().norm();

  // get information about sparse matrix
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  const unsigned nnz = lse.A().nonZeros();

  // get squared norms of the rows of the matrix (precompute for performance)

  std::vector<double> h_sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    // get the row i of the matrix
    Eigen::SparseVector<double> A_row_i = lse.A().innerVector(i);
    h_sq_norms[i] = A_row_i.dot(A_row_i);

    if (h_sq_norms[i] < 1e-7) {
      return KaczmarzSolverStatus::ZeroNormRow;  // check for zero norm rows
    }
  }

  // get maximum nr of nnz in row
  int max_nnz_in_row = 0;
  int nnz_in_row = 0;  // preallocate
  for (unsigned i = 0; i < rows; i++) {
    nnz_in_row = A_outer[i + 1] - A_outer[i];
    max_nnz_in_row = std::max(max_nnz_in_row, nnz_in_row);
  }

  // call carp solver for beginning
  return invoke_carp_solver_gpu(A_outer, A_inner, A_values, b, x.data(),
                                h_sq_norms.data(), rows, cols, nnz,
                                max_iterations, precision, max_nnz_in_row, b_norm);
}