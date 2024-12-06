#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <iostream>
#include "cusolver.hpp"


KaczmarzSolverStatus cusolver(const SparseLinearSystem& lse, Vector& x,
                                        const unsigned max_iterations,
                                        const double precision){
//create handel
cusolverSpHandle_t handle;
cusolverSpCreate(&handle);

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
  assert(cols == rows);  // check for square matrix, always if Galerkin
  const unsigned dim = cols;
  const unsigned nnz = lse.A().nonZeros();

  //create description for A
  cusparseMatDescr_t descrA;
cusparseCreateMatDescr(&descrA);
cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

int singularity;


cusolverSpDcsrlsvqr(handle,
                 dim,
                 nnz,
                 descrA,
                 A_values,
                 A_outer,
                 A_inner,
                 b,
                 precision,
                 0,
                 x.data(),
                 &singularity);

                 if (singularity >= 0) {
        std::cout << "Matrix is singular at row " << singularity << std::endl;
    }

    cusolverSpDestroy(handle);

    return KaczmarzSolverStatus::Converged;


}