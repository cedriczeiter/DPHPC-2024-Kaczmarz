#include "basic_cuda.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>


#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(err); \
    } \
  } while (0)



__global__ void kaczmarz_dense_update(double *x, const double *A, const double *b,
                                      const unsigned rows, const unsigned cols,
                                      double *row_sq_norms) {
  const unsigned row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx < rows) {
    const double *const a_row = A + row_idx * cols;
    double dot_product = 0.0;
    double row_sq_norm = 0.0;

    // Compute the dot product and squared norm
    for (unsigned j = 0; j < cols; j++) {
      dot_product += a_row[j] * x[j];
      row_sq_norm += a_row[j] * a_row[j];
    }

    // Store the row squared norm
    row_sq_norms[row_idx] = row_sq_norm;

    // Stop if the squared norm of the row is zero
    if (row_sq_norm < 1e-10) {
      return;
    }

    const double correction = (b[row_idx] - dot_product) / row_sq_norm;
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }
  }
}


double invoke_dense_kaczmarz_update(const DenseLinearSystem &lse, double *x, const unsigned rows, const unsigned cols) {
  const unsigned thread_count = 64; // Use a reasonable number of threads per block
  CUDA_CHECK(cudaDeviceReset());
  // Copying memory to the GPU
  const auto gpu_malloc_and_copy = [](const double *v, const size_t byte_count) {
    double *gpu_memory;
    CUDA_CHECK(cudaMalloc(&gpu_memory, byte_count));
    CUDA_CHECK(cudaMemcpy(gpu_memory, v, byte_count, cudaMemcpyHostToDevice));
    return gpu_memory;
  };
  double *x_gpu = gpu_malloc_and_copy(x, cols * sizeof(double));
  double *A_gpu = gpu_malloc_and_copy(lse.A(), rows * cols * sizeof(double));
  double *b_gpu = gpu_malloc_and_copy(lse.b(), rows * sizeof(double));
  // Allocate memory for row squared norms on the GPU
  double *row_sq_norms_gpu;
  CUDA_CHECK(cudaMalloc(&row_sq_norms_gpu, rows * sizeof(double)));
  // Launch the kernel
  kaczmarz_dense_update<<<(rows + thread_count - 1) / thread_count, thread_count>>>(
      x_gpu, A_gpu, b_gpu, rows, cols, row_sq_norms_gpu);
  CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors
  // Copy the updated x back to the host
  CUDA_CHECK(cudaMemcpy(x, x_gpu, cols * sizeof(double), cudaMemcpyDeviceToHost));
  // Copy the row squared norms back to the host
  std::vector<double> row_sq_norms(rows);
  CUDA_CHECK(cudaMemcpy(row_sq_norms.data(), row_sq_norms_gpu, rows * sizeof(double), cudaMemcpyDeviceToHost));
  // Free GPU memory
  CUDA_CHECK(cudaFree(x_gpu));
  CUDA_CHECK(cudaFree(A_gpu));
  CUDA_CHECK(cudaFree(b_gpu));
  CUDA_CHECK(cudaFree(row_sq_norms_gpu));
  CUDA_CHECK(cudaDeviceReset());
  // Find and return the smallest row squared norm
  double smallest_row_sq_norm = *std::min_element(row_sq_norms.begin(), row_sq_norms.end());
  return smallest_row_sq_norm;
}