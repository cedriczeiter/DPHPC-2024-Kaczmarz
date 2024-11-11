#include "cuda_funcs.hpp"

#include <cassert>

// write cuda function that takes a matrix and a vector and calculates dot prodcut of every row with the vector added up
__global__ void dot_product_kernel(const double *A, const double *x, double *result, const unsigned rows, const unsigned cols)
{
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double dot_product = 0.0;
        for (unsigned j = 0; j < cols; j++)
        {
            dot_product += A[row * cols + j] * x[j];
        }
        result[row] = dot_product;
    }
}

void dot_product_cuda(const double *h_A, const double *h_x, double *h_result, const unsigned rows, const unsigned cols)
{
    double *d_A, *d_x, *d_result;
    cudaMalloc(&d_A, rows * cols * sizeof(double));
    cudaMalloc(&d_x, cols * sizeof(double));
    cudaMalloc(&d_result, rows * sizeof(double));

    cudaMemcpy(d_A, h_A, rows * cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(double), cudaMemcpyHostToDevice);

    dot_product_kernel<<<1, rows>>>(d_A, d_x, d_result, rows, cols);

    cudaMemcpy(h_result, d_result, rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_result);
}

// write cuda function that takes a matrix and a vector and calculates squared norm of every row
__global__ void squared_norm_kernel(const double *A, double *result, const unsigned rows, const unsigned cols)
{
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        double row_sq_norm = 0.0;
        for (unsigned j = 0; j < cols; j++)
        {
            row_sq_norm += A[row * cols + j] * A[row * cols + j];
        }
        result[row] = row_sq_norm;
    }
}

void squared_norm_cuda(const double *h_A, double *h_result, const unsigned rows, const unsigned cols)
{
    double *d_A, *d_result;
    cudaMalloc(&d_A, rows * cols * sizeof(double));
    cudaMalloc(&d_result, rows * sizeof(double));

    cudaMemcpy(d_A, h_A, rows * cols * sizeof(double), cudaMemcpyHostToDevice);

    squared_norm_kernel<<<1, rows>>>(d_A, d_result, rows, cols);

    cudaMemcpy(h_result, d_result, rows * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_result);
}