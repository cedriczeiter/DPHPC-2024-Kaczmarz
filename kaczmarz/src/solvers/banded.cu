#include "banded_cuda.hpp"

/**
 * Expects x and A_data padded so that edge cases need not be dealt with.
 */
__global__ void kaczmarz_banded_update(double *x_gl, double *A_data_gl,
                                       double *sq_norms_gl, double *b_gl,
                                       int bandwidth, const int dim) {
  bandwidth = 2;

  extern __shared__ double s[];
  double *x = s + bandwidth;
  double *A_data = &s[dim + 2 * bandwidth];
  double *sq_norms = &A_data[(2 * bandwidth + 1) * dim];
  double *b = &sq_norms[dim];

  if (threadIdx.x == 0) {
    for (int i = -bandwidth; i < dim + bandwidth; i++) {
      x[i] = x_gl[i];
    }
    for (int i = 0; i < (2 * bandwidth + 1) * dim; i++) {
      A_data[i] = A_data_gl[i];
    }
    for (int i = 0; i < dim; i++) {
      sq_norms[i] = sq_norms_gl[i];
    }
    for (int i = 0; i < dim; i++) {
      b[i] = b_gl[i];
    }
  }

  for (unsigned iter = 0; iter < 1'000'000; iter++) {
    for (int row_i = 0; row_i < 2 * bandwidth + 1; row_i++) {
      const int row_idx = threadIdx.x * (2 * bandwidth + 1) + row_i;
      double dot = 0.0;
      for (int i = 0; i < 2 * bandwidth + 1; i++) {
        dot += A_data[(2 * bandwidth + 1) * row_idx + i] *
               x[row_idx - bandwidth + i];
      }
      const double update_coeff = (b[row_idx] - dot) / sq_norms[row_idx];
      for (int i = 0; i < 2 * bandwidth + 1; i++) {
        x[row_idx - bandwidth + i] +=
            update_coeff * A_data[(2 * bandwidth + 1) * row_idx + i];
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    for (int i = -bandwidth; i < dim + bandwidth; i++) {
      x_gl[i] = x[i];
    }
    for (int i = 0; i < (2 * bandwidth + 1) * dim; i++) {
      A_data_gl[i] = A_data[i];
    }
    for (int i = 0; i < dim; i++) {
      sq_norms_gl[i] = sq_norms[i];
    }
    for (int i = 0; i < dim; i++) {
      b_gl[i] = b[i];
    }
  }
}

void invoke_kaczmarz_banded_update(const unsigned bandwidth, const unsigned dim,
                                   const unsigned thread_count,
                                   const std::vector<double> &A_data_padded,
                                   std::vector<double> &x_padded,
                                   const std::vector<double> &sq_norms_padded,
                                   const std::vector<double> &b_padded) {
  // copying memory to the GPU
  const auto gpu_malloc_and_copy = [](const std::vector<double> &v) {
    double *gpu_memory;
    const size_t byte_count = v.size() * sizeof(double);
    cudaMalloc(&gpu_memory, byte_count);
    cudaMemcpy(gpu_memory, &v[0], byte_count, cudaMemcpyHostToDevice);
    return gpu_memory;
  };
  double *x_gpu = gpu_malloc_and_copy(x_padded);
  double *A_data_gpu = gpu_malloc_and_copy(A_data_padded);
  double *sq_norms_gpu = gpu_malloc_and_copy(sq_norms_padded);
  double *b_gpu = gpu_malloc_and_copy(b_padded);
  const size_t shared_memory_size =
      sizeof(double) * (x_padded.size() + A_data_padded.size() +
                        sq_norms_padded.size() + b_padded.size());
  kaczmarz_banded_update<<<1, thread_count, shared_memory_size>>>(
      x_gpu + bandwidth, A_data_gpu, sq_norms_gpu, b_gpu, bandwidth, dim);
  cudaMemcpy(&x_padded[0], x_gpu, x_padded.size() * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(x_gpu);
  cudaFree(A_data_gpu);
  cudaFree(sq_norms_gpu);
  cudaFree(b_gpu);
}
