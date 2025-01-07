#include <cooperative_groups.h>
#include <stdio.h>

#include "nolpde_cuda.hpp"

namespace cg = cooperative_groups;

__global__ void nolpde(double *x, const unsigned *A_outer,
                       const unsigned *A_inner, const double *A_values,
                       const unsigned *block_boundaries, const double *sq_norms,
                       const double *b, unsigned iterations) {
  cg::grid_group grid = cg::this_grid();
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned stage = 0; stage < 4; stage++) {
      const unsigned row_idx_from = block_boundaries[4 * tid + stage];
      const unsigned row_idx_to = block_boundaries[4 * tid + stage + 1];
      for (unsigned row_idx = row_idx_from; row_idx < row_idx_to; row_idx++) {
        double dot = 0.0;
        const unsigned A_idx_from = A_outer[row_idx];
        const unsigned A_idx_to = A_outer[row_idx + 1];
        for (unsigned i = A_idx_from; i < A_idx_to; i++) {
          dot += A_values[i] * x[A_inner[i]];
        }
        const double update_coeff = (b[row_idx] - dot) / sq_norms[row_idx];
        for (unsigned i = A_idx_from; i < A_idx_to; i++) {
          x[A_inner[i]] += update_coeff * A_values[i];
        }
      }
      grid.sync();
    }
  }
}

void invoke_nolpde_kernel(unsigned block_count, unsigned threads_per_block,
                          double *x, const unsigned *A_outer,
                          const unsigned *A_inner, const double *A_values,
                          const unsigned *block_boundaries,
                          const double *sq_norms, const double *b,
                          unsigned iterations) {
  void *kernel_args[] = {
      (void *)&x,        (void *)&A_outer,          (void *)&A_inner,
      (void *)&A_values, (void *)&block_boundaries, (void *)&sq_norms,
      (void *)&b,        (void *)&iterations};

  cudaError_t err = cudaLaunchCooperativeKernel((void *)nolpde, block_count,
                                                threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}
