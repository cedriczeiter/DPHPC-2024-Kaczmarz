#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <stdexcept>
#include "cuda_native.hpp"

KaczmarzSolverStatus native_cuda_solver(const SparseLinearSystem& lse, Vector& x,
                                        const unsigned max_iterations,
                                        const double precision) {
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

    cudaMalloc((void**)&d_rowPtr, sizeof(int) * (rows + 1));
    cudaMalloc((void**)&d_colInd, sizeof(int) * nnz);
    cudaMalloc((void**)&d_values, sizeof(double) * nnz);
    cudaMalloc((void**)&d_b, sizeof(double) * rows);
    cudaMalloc((void**)&d_x, sizeof(double) * cols);

    // Copy data to device
    cudaMemcpy(d_rowPtr, rowPtr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colInd, colInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values, sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * rows, cudaMemcpyHostToDevice);

    // Initialize cuSolverSP handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusolverStatus_t status = cusolverSpCreate(&cusolverH);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_rowPtr);
        cudaFree(d_colInd);
        cudaFree(d_values);
        cudaFree(d_b);
        cudaFree(d_x);
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
cusolverStatus_t status;

// Call the cuSolver function
status = cusolverSpDcsrlsvchol(
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
    if (status != CUSOLVER_STATUS_SUCCESS || singularity >= 0) {
        cusolverSpDestroy(cusolverH);
        cudaFree(d_rowPtr);
        cudaFree(d_colInd);
        cudaFree(d_values);
        cudaFree(d_b);
        cudaFree(d_x);
        return KaczmarzSolverStatus::OutOfIterations;
    }

    // Copy result back to host
    cudaMemcpy(x.data(), d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost);

    // Free resources
    cusolverSpDestroy(cusolverH);
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_values);
    cudaFree(d_b);
    cudaFree(d_x);

    return KaczmarzSolverStatus::Converged;
}