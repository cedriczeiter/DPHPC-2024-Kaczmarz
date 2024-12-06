#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <iostream>
#include "cusolver.hpp"
#include "carp_utils.hpp"


KaczmarzSolverStatus cusolver(const SparseLinearSystem& lse, Vector& x,
                                        const unsigned max_iterations,
                                        const double precision){
// Extract matrix data in CSR format
    const auto& A = lse.A();  // Eigen::SparseMatrix
    const auto& b = lse.b();  // Eigen::VectorXd

    const unsigned rows = A.rows();
    const unsigned cols = A.cols();
    const int nnz = A.nonZeros();  // Number of non-zero entries

    // Get CSR pointers from Eigen
    const int* rowPtr = A.outerIndexPtr();  // Row pointers
    const int* colInd = A.innerIndexPtr();  // Column indices
    const double* values = A.valuePtr();    // Non-zero values

    // Allocate device memory
    int *d_rowPtr, *d_colInd;
    double *d_values, *d_b, *d_x;

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_rowPtr, sizeof(int) * (rows + 1)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_colInd, sizeof(int) * nnz));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_values, sizeof(double) * nnz));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_b, sizeof(double) * rows));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_x, sizeof(double) * cols));

    // Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(d_rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_colInd, colInd, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_values, values, sizeof(double) * nnz, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b, b.data(), sizeof(double) * rows, cudaMemcpyHostToDevice));

    // Initialize cuSolverSP handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusolverStatus_t status = cusolverSpCreate(&cusolverH);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        CUDA_SAFE_CALL(cudaFree(d_rowPtr));
        CUDA_SAFE_CALL(cudaFree(d_colInd));
        CUDA_SAFE_CALL(cudaFree(d_values));
        CUDA_SAFE_CALL(cudaFree(d_b));
        CUDA_SAFE_CALL(cudaFree(d_x));
        throw std::runtime_error("Failed to create cuSolverSP handle.");
    }

    // Solve the system using Cholesky factorization
    int singularity = 0;

// Declare and initialize the matrix descriptor
cusparseMatDescr_t descrA;
cusparseCreateMatDescr(&descrA);
cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

// Declare cusolver status variable (once)
cusolverStatus_t status_solver;

// Call the cuSolver function
status_solver = cusolverSpDcsrlsvqr(
    cusolverH,          // cuSolver handle
    rows,               // Number of rows
    nnz,                // Number of non-zero elements
    descrA,             // Matrix descriptor
    d_values,           // Matrix values (double*)
    d_rowPtr,           // Row pointers (int*)
    d_colInd,           // Column indices (int*)
    d_b,                // Right-hand side vector (double*)
    precision,          // Tolerance
    0,                  // Reorder flag
    d_x,                // Solution vector (double*)
    &singularity        // Singular matrix info
);
    // Check solver status
    if (status_solver != CUSOLVER_STATUS_SUCCESS || singularity >= 0) {
        cusolverSpDestroy(cusolverH);
        cudaFree(d_rowPtr);
        cudaFree(d_colInd);
        cudaFree(d_values);
        cudaFree(d_b);
        cudaFree(d_x);
        return KaczmarzSolverStatus::OutOfIterations;
    }

    // Copy result back to host
    CUDA_SAFE_CALL(cudaMemcpy(x.data(), d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost));

    // Free resources
    cusolverSpDestroy(cusolverH);
    CUDA_SAFE_CALL(cudaFree(d_rowPtr));
    CUDA_SAFE_CALL(cudaFree(d_colInd));
    CUDA_SAFE_CALL(cudaFree(d_values));
    CUDA_SAFE_CALL(cudaFree(d_b));
    CUDA_SAFE_CALL(cudaFree(d_x));

    return KaczmarzSolverStatus::Converged;
}