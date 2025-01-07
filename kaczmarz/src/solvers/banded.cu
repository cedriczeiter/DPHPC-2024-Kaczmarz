#include <cooperative_groups.h>
#include <stdio.h>

#include "banded_cuda.hpp"

namespace cg = cooperative_groups;

/**
 * Expects x and A_data padded so that edge cases need not be dealt with.
 */
__global__ void kaczmarz_banded_grouping1(
    double *__restrict__ x, const double *__restrict__ A_data,
    const double *__restrict__ sq_norms, const double *__restrict__ b,
    const unsigned iterations, int bandwidth, const unsigned rows_per_group,
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
    }
    grid.sync();
  }
}

void invoke_banded_grouping1_kernel(
    const unsigned block_count, const unsigned threads_per_block,
    double *const x, const double *const A_data, const double *const sq_norms,
    const double *const b, const unsigned iterations, const int bandwidth,
    const unsigned rows_per_group, const unsigned extra_rows) {
  void *kernel_args[] = {(void *)&x,
                         (void *)&A_data,
                         (void *)&sq_norms,
                         (void *)&b,
                         (void *)&iterations,
                         (void *)&bandwidth,
                         (void *)&rows_per_group,
                         (void *)&extra_rows};

  cudaError_t err =
      cudaLaunchCooperativeKernel((void *)kaczmarz_banded_grouping1,
                                  block_count, threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}

__global__ void kaczmarz_banded_grouping2(
    double *__restrict__ x, const double *__restrict__ A_data,
    const double *__restrict__ sq_norms, const double *__restrict__ b,
    const unsigned iterations, int bandwidth, const unsigned rows_per_thread,
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
      }
      grid.sync();
    }
  }
}

void invoke_banded_grouping2_kernel(
    const unsigned block_count, const unsigned threads_per_block,
    double *const x, const double *const A_data, const double *const sq_norms,
    const double *const b, const unsigned iterations, const int bandwidth,
    const unsigned rows_per_thread, const unsigned extra_rows) {
  void *kernel_args[] = {(void *)&x,
                         (void *)&A_data,
                         (void *)&sq_norms,
                         (void *)&b,
                         (void *)&iterations,
                         (void *)&bandwidth,
                         (void *)&rows_per_thread,
                         (void *)&extra_rows};

  cudaError_t err =
      cudaLaunchCooperativeKernel((void *)kaczmarz_banded_grouping2,
                                  block_count, threads_per_block, kernel_args);

  if (err != cudaSuccess) {
    printf("Error launching kernel: %s\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
}
