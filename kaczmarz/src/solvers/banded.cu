#include "banded_cuda.hpp"

#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

/**
 * Expects x and A_data padded so that edge cases need not be dealt with.
 */
__global__ void kaczmarz_banded_update(double *x, double *A_data,
                                       double *sq_norms, double *b,
                                       int bandwidth, unsigned width,
                                       unsigned threads_per_block) {
  bandwidth = 2;
  //width = 5;
  cg::grid_group grid = cg::this_grid();
  const unsigned thread_id = blockIdx.x * threads_per_block + threadIdx.x;
  for (unsigned iter = 0; iter < 10'000'000; iter++) {
    for (unsigned row_i = 0; row_i < width; row_i++) {
      const int row_idx = thread_id * 2 * width + row_i;
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
      for (int i = 0; i < 2 * bandwidth + 1; i++) {
        dot += A_data[(2 * bandwidth + 1) * row_idx + i] *
               x[row_idx - bandwidth + i];
      }
    }
    grid.sync();
    //__syncthreads();
    for (unsigned row_i = 0; row_i < width; row_i++) {
      const int row_idx = thread_id * 2 * width + width + row_i;
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
      for (int i = 0; i < 2 * bandwidth + 1; i++) {
        dot += A_data[(2 * bandwidth + 1) * row_idx + i] *
               x[row_idx - bandwidth + i];
      }
    }
    grid.sync();
    //__syncthreads();
  }
}

void invoke_kaczmarz_banded_update(const unsigned bandwidth,
                                   const unsigned threads_per_block,
                                   const unsigned block_count,
                                   const unsigned width,
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

  //const dim3 grid_dim(block_count);
  //const dim3 block_dim(threads_per_block);

  double *x_gpu_arg = x_gpu + bandwidth;

  // Kernel arguments
  void *kernel_args[] = {
      &x_gpu_arg,
      &A_data_gpu,
      &sq_norms_gpu,
      &b_gpu,
      (void*)&bandwidth,
      (void*)&width,
      (void*)&threads_per_block,
  };

  // Launch the cooperative kernel
  cudaError_t err = cudaLaunchCooperativeKernel(
      (void *)kaczmarz_banded_update, block_count, threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();


  /*
  kaczmarz_banded_update<<<block_count, threads_per_block>>>(x_gpu + bandwidth, A_data_gpu,
                                              sq_norms_gpu, b_gpu, bandwidth, width, threads_per_block);
                                              */
  cudaMemcpy(&x_padded[0], x_gpu, x_padded.size() * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(x_gpu);
  cudaFree(A_data_gpu);
  cudaFree(sq_norms_gpu);
  cudaFree(b_gpu);
}
