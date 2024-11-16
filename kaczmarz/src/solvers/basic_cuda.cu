#include "basic_cuda.hpp"

__global__ void kaczmarz_dense_update(double *x, double *A, double *sq_norms,
                                      double *b, const unsigned rows,
                                      const unsigned cols) {
  const unsigned row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_idx < rows) {
    const double *const a_row = A + row_idx * cols;
    double dot_product = 0.0;
    double row_sq_norm = sq_norms[row_idx];

    // Compute the dot product
    for (unsigned j = 0; j < cols; j++) {
      dot_product += a_row[j] * x[j];
    }

    // Stop if a row squared norm of a row is zero
    if (row_sq_norm < 1e-10) {
      row_sq_norm = 0;
    }

    const double correction = (b[row_idx] - dot_product) / row_sq_norm;
    for (unsigned j = 0; j < cols; j++) {
      x[j] += a_row[j] * correction;
    }
  }
}

void invoke_dense_kaczmarz_update(const unsigned thread_count,
                                  const DenseLinearSystem &lse, double *x,
                                  const unsigned rows, const unsigned cols) {
  // Copying memory to the GPU
  const auto gpu_malloc_and_copy = [](const double *v, const size_t byte_count) {
    double *gpu_memory;
    cudaMalloc(&gpu_memory, byte_count);
    cudaMemcpy(gpu_memory, v, byte_count, cudaMemcpyHostToDevice);
    return gpu_memory;
  };
  double *x_gpu = gpu_malloc_and_copy(x, cols * sizeof(double));
  double *A_gpu = gpu_malloc_and_copy(lse.A(), rows * cols * sizeof(double));
  double *sq_norms_gpu = gpu_malloc_and_copy(lse.sq_norms(), rows * sizeof(double));
  double *b_gpu = gpu_malloc_and_copy(lse.b(), rows * sizeof(double);

  kaczmarz_dense_update<<<(rows + thread_count - 1) / thread_count, thread_count>>>(
      x_gpu, A_gpu, sq_norms_gpu, b_gpu, rows, cols);

  cudaMemcpy(x, x_gpu, cols * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(x_gpu);
  cudaFree(A_gpu);
  cudaFree(sq_norms_gpu);
  cudaFree(b_gpu);
}