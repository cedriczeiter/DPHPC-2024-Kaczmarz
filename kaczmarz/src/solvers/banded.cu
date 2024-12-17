#include <cooperative_groups.h>
#include <stdio.h>

#include <cassert>

#include "banded_cuda.hpp"

namespace cg = cooperative_groups;

/**
 * Expects x and A_data padded so that edge cases need not be dealt with.
 */
__global__ void kaczmarz_banded_grouping1(double *x, double *A_data,
                                          double *sq_norms, double *b,
                                          const unsigned iterations,
                                          int bandwidth,
                                          const unsigned rows_per_group,
                                          const unsigned extra_rows) {
  bandwidth = 2;
  cg::grid_group grid = cg::this_grid();
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned start_row_idx =
      (2 * tid) * rows_per_group + min(2 * tid, extra_rows);
  const unsigned mid_row_idx =
      (2 * tid + 1) * rows_per_group + min(2 * tid + 1, extra_rows);
  const unsigned end_row_idx =
      (2 * tid + 2) * rows_per_group + min(2 * tid + 2, extra_rows);
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (int row_idx = start_row_idx; row_idx < (int)mid_row_idx; row_idx++) {
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
    for (int row_idx = mid_row_idx; row_idx < (int)end_row_idx; row_idx++) {
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
  }
}

template <typename T>
T *gpu_malloc_and_copy(const std::vector<T> &v) {
  T *gpu_memory;
  const size_t byte_count = v.size() * sizeof(T);
  cudaMalloc(&gpu_memory, byte_count);
  cudaMemcpy(gpu_memory, &v[0], byte_count, cudaMemcpyHostToDevice);
  return gpu_memory;
}

void invoke_kaczmarz_banded_cuda_grouping1(UnpackedBandedSystem &sys,
                                           const unsigned iterations,
                                           const unsigned block_count,
                                           const unsigned threads_per_block) {
  // copying memory to the GPU
  double *x_gpu = gpu_malloc_and_copy(sys.x_padded);
  double *A_data_gpu = gpu_malloc_and_copy(sys.A_data);
  double *sq_norms_gpu = gpu_malloc_and_copy(sys.sq_norms);
  double *b_gpu = gpu_malloc_and_copy(sys.b);

  double *x_gpu_arg = x_gpu + sys.bandwidth;

  const unsigned group_count = 2 * block_count * threads_per_block;

  unsigned rows_per_group_arg = sys.dim / group_count;
  unsigned extra_rows_arg = sys.dim % group_count;

  // kernel arguments
  void *kernel_args[] = {&x_gpu_arg,
                         &A_data_gpu,
                         &sq_norms_gpu,
                         &b_gpu,
                         (void *)&iterations,
                         (void *)&sys.bandwidth,
                         (void *)&rows_per_group_arg,
                         (void *)&extra_rows_arg};

  // launch the cooperative kernel
  cudaError_t err =
      cudaLaunchCooperativeKernel((void *)kaczmarz_banded_grouping1,
                                  block_count, threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  cudaMemcpy(&sys.x_padded[0], x_gpu, sys.x_padded.size() * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(x_gpu);
  cudaFree(A_data_gpu);
  cudaFree(sq_norms_gpu);
  cudaFree(b_gpu);
}

__global__ void kaczmarz_banded_grouping2(double *x, double *A_data,
                                          double *sq_norms, double *b,
                                          const unsigned iterations,
                                          int bandwidth,
                                          const unsigned rows_per_thread,
                                          const unsigned extra_rows) {
  bandwidth = 2;
  cg::grid_group grid = cg::this_grid();
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned row_in_group_from =
      tid * rows_per_thread + min(tid, extra_rows);
  const unsigned row_in_group_to =
      (tid + 1) * rows_per_thread + min(tid + 1, extra_rows);
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned group_idx = 0; group_idx < 2 * (unsigned)bandwidth + 1;
         group_idx++) {
      for (unsigned row_in_group_idx = row_in_group_from;
           row_in_group_idx < row_in_group_to; row_in_group_idx++) {
        const int row_idx = row_in_group_idx * (2 * bandwidth + 1) + group_idx;
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
    }
  }
}

void invoke_kaczmarz_banded_cuda_grouping2(UnpackedBandedSystem &sys,
                                           const unsigned iterations,
                                           const unsigned block_count,
                                           const unsigned threads_per_block) {
  // copying memory to the GPU
  double *x_gpu = gpu_malloc_and_copy(sys.x_padded);
  double *A_data_gpu = gpu_malloc_and_copy(sys.A_data);
  double *sq_norms_gpu = gpu_malloc_and_copy(sys.sq_norms);
  double *b_gpu = gpu_malloc_and_copy(sys.b);

  double *x_gpu_arg = x_gpu + sys.bandwidth;

  assert(sys.dim % (2 * sys.bandwidth + 1) == 0);

  const unsigned rows_per_group = sys.dim / (2 * sys.bandwidth + 1);

  const unsigned thread_count = block_count * threads_per_block;

  unsigned rows_per_thread_arg = rows_per_group / thread_count;
  unsigned extra_rows_arg = rows_per_group % thread_count;

  // kernel arguments
  void *kernel_args[] = {&x_gpu_arg,
                         &A_data_gpu,
                         &sq_norms_gpu,
                         &b_gpu,
                         (void *)&iterations,
                         (void *)&sys.bandwidth,
                         (void *)&rows_per_thread_arg,
                         (void *)&extra_rows_arg};

  // launch the cooperative kernel
  cudaError_t err =
      cudaLaunchCooperativeKernel((void *)kaczmarz_banded_grouping2,
                                  block_count, threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();

  cudaMemcpy(&sys.x_padded[0], x_gpu, sys.x_padded.size() * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(x_gpu);
  cudaFree(A_data_gpu);
  cudaFree(sq_norms_gpu);
  cudaFree(b_gpu);
}
