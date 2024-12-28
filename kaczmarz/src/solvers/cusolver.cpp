#include "cusolver.hpp"

#include <cuda_runtime.h>
#include <cusolverSp.h>

#include <iostream>

#include "carp_utils.hpp"
#include "cudss.h"

#define CUDSS_EXAMPLE_FREE   \
  do {                       \
    cudaFree(csr_offsets_d); \
    cudaFree(csr_columns_d); \
    cudaFree(csr_values_d);  \
    cudaFree(x_values_d);    \
    cudaFree(b_values_d);    \
  } while (0);

#define CUDA_CALL_AND_CHECK(call, msg)                                      \
  do {                                                                      \
    cuda_error = call;                                                      \
    if (cuda_error != cudaSuccess) {                                        \
      printf("Example FAILED: CUDA API returned error = %d, details: " #msg \
             "\n",                                                          \
             cuda_error);                                                   \
      CUDSS_EXAMPLE_FREE;                                                   \
      return KaczmarzSolverStatus::OutOfIterations;                                                            \
    }                                                                       \
  } while (0);

#define CUDSS_CALL_AND_CHECK(call, status, msg)                                \
  do {                                                                         \
    status = call;                                                             \
    if (status != CUDSS_STATUS_SUCCESS) {                                      \
      printf(                                                                  \
          "Example FAILED: CUDSS call ended unsuccessfully with status = %d, " \
          "details: " #msg "\n",                                               \
          status);                                                             \
      CUDSS_EXAMPLE_FREE;                                                      \
      return KaczmarzSolverStatus::OutOfIterations;                                                               \
    }                                                                          \
  } while (0);

KaczmarzSolverStatus cusolver(const SparseLinearSystem& lse, Vector& x,
                              const unsigned max_iterations,
                              const double precision) {
  cudaError_t cuda_error = cudaSuccess;
  cudssStatus_t status = CUDSS_STATUS_SUCCESS;

  // Extract matrix data in CSR format
  const auto& A = lse.A();  // Eigen::SparseMatrix
  const auto& b = lse.b();  // Eigen::VectorXd

  const unsigned n = A.rows();
  const unsigned cols = A.cols();
  const int nnz = A.nonZeros();  // Number of non-zero entries
  const unsigned nrhs = 1;

  int* csr_offsets_d = NULL;
  int* csr_columns_d = NULL;
  double* csr_values_d = NULL;
  double *x_values_d = NULL, *b_values_d = NULL;

  /* Allocate device memory for A, x and b */
  CUDA_CALL_AND_CHECK(cudaMalloc(&csr_offsets_d, (n + 1) * sizeof(int)),
                      "cudaMalloc for csr_offsets");
  CUDA_CALL_AND_CHECK(cudaMalloc(&csr_columns_d, nnz * sizeof(int)),
                      "cudaMalloc for csr_columns");
  CUDA_CALL_AND_CHECK(cudaMalloc(&csr_values_d, nnz * sizeof(double)),
                      "cudaMalloc for csr_values");
  CUDA_CALL_AND_CHECK(cudaMalloc(&b_values_d,  n * sizeof(double)),
                      "cudaMalloc for b_values");
  CUDA_CALL_AND_CHECK(cudaMalloc(&x_values_d, n * sizeof(double)),
                      "cudaMalloc for x_values");

  /* Copy host memory to device for A and b */
  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_offsets_d, A.outerIndexPtr(),
                                 (n + 1) * sizeof(int), cudaMemcpyHostToDevice),
                      "cudaMemcpy for csr_offsets");
  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_columns_d, A.innerIndexPtr(),
                                 nnz * sizeof(int), cudaMemcpyHostToDevice),
                      "cudaMemcpy for csr_columns");
  CUDA_CALL_AND_CHECK(cudaMemcpy(csr_values_d, A.valuePtr(),
                                 nnz * sizeof(double), cudaMemcpyHostToDevice),
                      "cudaMemcpy for csr_values");
  CUDA_CALL_AND_CHECK(
      cudaMemcpy(b_values_d, b.data(), n * sizeof(double),
                 cudaMemcpyHostToDevice),
      "cudaMemcpy for b_values");

  /* Create a CUDA stream */
  cudaStream_t stream = NULL;
  CUDA_CALL_AND_CHECK(cudaStreamCreate(&stream), "cudaStreamCreate");

  /* Creating the cuDSS library handle */
  cudssHandle_t handle;

  CUDSS_CALL_AND_CHECK(cudssCreate(&handle), status, "cudssCreate");

  /* (optional) Setting the custom stream for the library handle */
  CUDSS_CALL_AND_CHECK(cudssSetStream(handle, stream), status,
                       "cudssSetStream");

  /* Creating cuDSS solver configuration and data objects */
  cudssConfig_t solverConfig;
  cudssData_t solverData;

  CUDSS_CALL_AND_CHECK(cudssConfigCreate(&solverConfig), status,
                       "cudssConfigCreate");
  CUDSS_CALL_AND_CHECK(cudssDataCreate(handle, &solverData), status,
                       "cudssDataCreate");

  /* Create matrix objects for the right-hand side b and solution x (as dense
   * matrices). */
  cudssMatrix_t x_matr, b_matr;

  int64_t nrows = n, ncols = n;
  int ldb = ncols, ldx = nrows;
  CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&b_matr, ncols, nrhs, ldb, b_values_d,
                                           CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                       status, "cudssMatrixCreateDn for b");
  CUDSS_CALL_AND_CHECK(cudssMatrixCreateDn(&x_matr, nrows, nrhs, ldx, x_values_d,
                                           CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR),
                       status, "cudssMatrixCreateDn for x");

  /* Create a matrix object for the sparse input matrix. */
  cudssMatrix_t A_matr;
  cudssMatrixType_t mtype = CUDSS_MTYPE_SPD;
  cudssMatrixViewType_t mview = CUDSS_MVIEW_UPPER;
  cudssIndexBase_t base = CUDSS_BASE_ZERO;
  CUDSS_CALL_AND_CHECK(
      cudssMatrixCreateCsr(&A_matr, nrows, ncols, nnz, csr_offsets_d, NULL,
                           csr_columns_d, csr_values_d, CUDA_R_32I, CUDA_R_64F,
                           mtype, mview, base),
      status, "cudssMatrixCreateCsr");

  /* Symbolic factorization */
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig,
                                    solverData, A_matr, x_matr, b_matr),
                       status, "cudssExecute for analysis");

  /* Factorization */
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION,
                                    solverConfig, solverData, A_matr, x_matr, b_matr),
                       status, "cudssExecute for factor");

  /* Solving */
  CUDSS_CALL_AND_CHECK(cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig,
                                    solverData, A_matr, x_matr, b_matr),
                       status, "cudssExecute for solve");

  /* Destroying opaque objects, matrix wrappers and the cuDSS library handle */
  CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(A_matr), status,
                       "cudssMatrixDestroy for A");
  CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(b_matr), status,
                       "cudssMatrixDestroy for b");
  CUDSS_CALL_AND_CHECK(cudssMatrixDestroy(x_matr), status,
                       "cudssMatrixDestroy for x");
  CUDSS_CALL_AND_CHECK(cudssDataDestroy(handle, solverData), status,
                       "cudssDataDestroy");
  CUDSS_CALL_AND_CHECK(cudssConfigDestroy(solverConfig), status,
                       "cudssConfigDestroy");
  CUDSS_CALL_AND_CHECK(cudssDestroy(handle), status, "cudssHandleDestroy");

  CUDA_CALL_AND_CHECK(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  /* Print the solution and compare against the exact solution */
  CUDA_CALL_AND_CHECK(
      cudaMemcpy(x_values_h, x_values_d, nrhs * n * sizeof(double),
                 cudaMemcpyDeviceToHost),
      "cudaMemcpy for x_values");

  if (status == CUDSS_STATUS_SUCCESS) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::ZeroNormRow;
}