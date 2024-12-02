#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>
#include <stdexcept>
#include "cuda_native.hpp"

KaczmarzSolverStatus native_cuda_solver(const SparseLinearSystem& lse, Vector& x,
                                        const unsigned max_iterations,
                                        const double precision) {
    // Extract matrix data in CSR format
    const auto A_csr = lse.A();  // CSR format matrix
    const auto b = lse.b();
    const unsigned rows = A_csr.rows();
    const unsigned cols = A_csr.cols();
    const int nnz = A_csr.nnz();

    // Allocate device memory
    int *d_A_outerIndex, *d_A_innerIndex;
    double *d_A_values, *d_b, *d_x;

    cudaMalloc((void**)&d_A_outerIndex, sizeof(int) * (rows + 1));
    cudaMalloc((void**)&d_A_innerIndex, sizeof(int) * nnz);
    cudaMalloc((void**)&d_A_values, sizeof(double) * nnz);
    cudaMalloc((void**)&d_b, sizeof(double) * rows);
    cudaMalloc((void**)&d_x, sizeof(double) * cols);

    // Copy data to device
    cudaMemcpy(d_A_outerIndex, A_csr.rowPtr(), sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_innerIndex, A_csr.colInd(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_csr.values(), sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * rows, cudaMemcpyHostToDevice);

    // Initialize cuSolverSP handle
    cusolverSpHandle_t cusolverH = nullptr;
    cusolverStatus_t status = cusolverSpCreate(&cusolverH);
    if (status != CUSOLVER_STATUS_SUCCESS) {
        cudaFree(d_A_outerIndex);
        cudaFree(d_A_innerIndex);
        cudaFree(d_A_values);
        cudaFree(d_b);
        cudaFree(d_x);
        throw std::runtime_error("Failed to create cuSolverSP handle.");
    }

    // Allocate workspace for cuSolverSP
    size_t workspaceSize = 0;
    cusolverSpDcsrlsvchol_bufferSize(cusolverH, rows, nnz, nullptr, d_A_outerIndex, d_A_innerIndex, d_A_values,
                                     nullptr, 0, &workspaceSize);

    void* d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspaceSize);

    // Solve the system using Cholesky factorization
    int singularity = 0;
    status = cusolverSpDcsrlsvchol(cusolverH, rows, nnz, nullptr, d_A_outerIndex, d_A_innerIndex, d_A_values, d_b,
                                   precision, max_iterations, d_x, &singularity);

    // Check solver status
    if (status != CUSOLVER_STATUS_SUCCESS || singularity >= 0) {
        cusolverSpDestroy(cusolverH);
        cudaFree(d_A_outerIndex);
        cudaFree(d_A_innerIndex);
        cudaFree(d_A_values);
        cudaFree(d_b);
        cudaFree(d_x);
        cudaFree(d_workspace);
        return KaczmarzSolverStatus::ZeroNormRow;
    }

    // Copy result back to host
    cudaMemcpy(x.data(), d_x, sizeof(double) * cols, cudaMemcpyDeviceToHost);

    // Free resources
    cusolverSpDestroy(cusolverH);
    cudaFree(d_A_outerIndex);
    cudaFree(d_A_innerIndex);
    cudaFree(d_A_values);
    cudaFree(d_b);
    cudaFree(d_x);
    cudaFree(d_workspace);

    return KaczmarzSolverStatus::Converged;
}