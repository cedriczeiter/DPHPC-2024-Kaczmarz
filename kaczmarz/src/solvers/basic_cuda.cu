#include "basic_cuda.hpp"

// Kernel to compute dot_product and row_sq_norm for a row
__global__ void computeRowSumsKernel(const double *A, const double *x, double *dot_product, double *row_sq_norm, int cols)
{

  // Get the right stuff onto GPU
  extern __shared__ double shared_mem[];

  double *partial_dot = shared_mem;
  double *partial_row_sq = shared_mem + blockDim.x;

  int tid = threadIdx.x;
  int idx = blockIdx.x * cols + tid;

  // Initialize the values that we work with
  partial_dot[tid] = 0.0;
  partial_row_sq[tid] = 0.0;

  // Compute the dot product and row square norm for every element in the row, one thread per element
  if (tid < cols)
  {
    double a_val = A[idx];
    partial_dot[tid] = a_val * x[tid];
    partial_row_sq[tid] = a_val * a_val; 
  }
  else //Handling of unused threads
  {
    partial_dot[tid] = 0.0;
    partial_row_sq[tid] = 0.0;
  }

  // Wait for all threads to finish
  __syncthreads();

  // Add up all the values with the stride pattern to get the final dot product and row square norm
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    if (tid < stride)
    {
      partial_dot[tid] += partial_dot[tid + stride];
      partial_row_sq[tid] += partial_row_sq[tid + stride];
    }
    __syncthreads();
  }

  // If tid is 0, write the final result to the output arrays
  if (tid == 0)
  {
    dot_product[blockIdx.x] = partial_dot[0];
    row_sq_norm[blockIdx.x] = partial_row_sq[0];
  }
}

// Wrapper function for invoking the kernel
bool computeRowSums(const std::vector<double> &A, const std::vector<double> &x,
                    std::vector<double> &dot_product, std::vector<double> &row_sq_norm,
                    int rows, int cols)
{
  double *d_A, *d_x, *d_dot_product, *d_row_sq_norm;

  // Allocate memory on the GPU
  cudaMalloc((void **)&d_A, rows * cols * sizeof(double));
  cudaMalloc((void **)&d_x, cols * sizeof(double));
  cudaMalloc((void **)&d_dot_product, rows * sizeof(double));
  cudaMalloc((void **)&d_row_sq_norm, rows * sizeof(double));

  cudaMemcpy(d_A, A.data(), rows * cols * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), cols * sizeof(double), cudaMemcpyHostToDevice);

  // Launch the kernel
  int threadsPerBlock = cols;
  computeRowSumsKernel<<<rows, threadsPerBlock, threadsPerBlock * 2 * sizeof(double)>>>(d_A, d_x, d_dot_product, d_row_sq_norm, cols);

  // Copy the results back to the host
  cudaMemcpy(dot_product.data(), d_dot_product, rows * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(row_sq_norm.data(), d_row_sq_norm, rows * sizeof(double), cudaMemcpyDeviceToHost);

  // Free the memory
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_dot_product);
  cudaFree(d_row_sq_norm);

  return cudaGetLastError() == cudaSuccess;
}