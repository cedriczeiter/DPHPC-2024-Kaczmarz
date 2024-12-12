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
                              const double precision, const double relaxation,
                              int& nr_of_steps) {
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

  const float b_norm = lse.b().norm();

  // get information about sparse matrix
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();
  assert(cols == rows);  // check for square matrix, always if Galerkin
  const unsigned dim = cols;
  const unsigned nnz = lse.A().nonZeros();

  // get squared norms of the rows of the matrix (precompute for performance)

  std::vector<float> h_sq_norms(dim);
  for (unsigned i = 0; i < dim; i++) {
    // get the row i of the matrix
    Eigen::SparseVector<double> A_row_i = lse.A().innerVector(i);
    h_sq_norms[i] = (float)A_row_i.dot(A_row_i);

    if (h_sq_norms[i] < 1e-7) {
      return KaczmarzSolverStatus::ZeroNormRow;  // check for zero norm rows
    }
  }

  // get maximum nr of nnz in row
  int max_nnz_in_row = 0;
  int nnz_in_row = 0;  // preallocate
  for (unsigned i = 0; i < dim; i++) {
    nnz_in_row = A_outer[i + 1] - A_outer[i];
    if (nnz_in_row > max_nnz_in_row) {
      max_nnz_in_row = nnz_in_row;
    }
  }
  std::cout << "Max nnz in a row: " << max_nnz_in_row << std::endl;

  float* float_x = new float[dim];
  for (int i = 0; i < dim; i++){
    float_x[i] = (float)x[i];
  }

  float* float_b = new float[dim];
  for (int i = 0; i < dim; i++){
    float_b[i] = (float)b[i];
  }

  float* float_A_values = new float[nnz];
  for (int i = 0; i < nnz; i++){
    float_A_values[i] = (float)A_values[i];
  }

  // call carp solver for beginning
  invoke_carp_solver_gpu(A_outer, A_inner, float_A_values, float_b, float_x,
                                h_sq_norms.data(), dim, nnz, max_iterations,
                                (float)precision, max_nnz_in_row, b_norm, nr_of_steps,
                                (float)relaxation);
  
  for (int i = 0; i < dim; i++){
    x[i] = (float)float_x[i];
  }
  return KaczmarzSolverStatus::Converged;
}