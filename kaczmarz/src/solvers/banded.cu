#include <cooperative_groups.h>
#include <stdio.h>

#include <cassert>

#include "banded_cuda.hpp"
#include "common.hpp"

namespace cg = cooperative_groups;

template <typename T>
static T *gpu_malloc_and_copy(const std::vector<T> &v) {
  T *gpu_memory;
  const size_t byte_count = v.size() * sizeof(T);
  cudaMalloc(&gpu_memory, byte_count);
  cudaMemcpy(gpu_memory, &v[0], byte_count, cudaMemcpyHostToDevice);
  return gpu_memory;
}

void GPUBandedSolver::setup(UnpackedBandedSystem *const sys) {
  this->cleanup();
  this->sys = sys;
  this->x_gpu = gpu_malloc_and_copy(sys->x_padded);
  this->A_data_gpu = gpu_malloc_and_copy(sys->A_data);
  this->sq_norms_gpu = gpu_malloc_and_copy(sys->sq_norms);
  this->b_gpu = gpu_malloc_and_copy(sys->b);
}

void GPUBandedSolver::flush_x() {
  const size_t byte_count = this->sys->x_padded.size() * sizeof(double);
  cudaMemcpy(&this->sys->x_padded[0], x_gpu, byte_count,
             cudaMemcpyDeviceToHost);
}

void GPUBandedSolver::cleanup() {
  if (this->x_gpu) {
    cudaFree(this->x_gpu);
    this->x_gpu = nullptr;
  }
  if (this->A_data_gpu) {
    cudaFree(this->A_data_gpu);
    this->A_data_gpu = nullptr;
  }
  if (this->sq_norms_gpu) {
    cudaFree(this->sq_norms_gpu);
    this->sq_norms_gpu = nullptr;
  }
  if (this->b_gpu) {
    cudaFree(this->b_gpu);
    this->b_gpu = nullptr;
  }
  this->sys = nullptr;
}

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

unsigned CUDAGrouping1BandedSolver::pad_dimension(const unsigned dim,
                                                  const unsigned bandwidth) {
  const unsigned thread_count = block_count * threads_per_block;
  return std::max(dim, bandwidth * 2 * 2 * thread_count);
}

void CUDAGrouping1BandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = this->sys->dim;
  const unsigned bandwidth = this->sys->bandwidth;

  double *x_gpu_arg = x_gpu + bandwidth;

  const unsigned group_count = 2 * block_count * threads_per_block;

  unsigned rows_per_group_arg = dim / group_count;
  unsigned extra_rows_arg = dim % group_count;

  // kernel arguments
  void *kernel_args[] = {&x_gpu_arg,
                         &A_data_gpu,
                         &sq_norms_gpu,
                         &b_gpu,
                         (void *)&iterations,
                         (void *)&bandwidth,
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
}

unsigned CUDAGrouping2BandedSolver::pad_dimension(const unsigned dim,
                                                  const unsigned bandwidth) {
  const unsigned group_count = 2 * bandwidth + 1;
  const unsigned rows_per_group = ceil_div(dim, group_count);
  return rows_per_group * group_count;
}

void CUDAGrouping2BandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = this->sys->dim;
  const unsigned bandwidth = this->sys->bandwidth;

  double *x_gpu_arg = this->x_gpu + bandwidth;

  assert(dim % (2 * bandwidth + 1) == 0);

  const unsigned rows_per_group = dim / (2 * bandwidth + 1);

  const unsigned thread_count = block_count * threads_per_block;

  unsigned rows_per_thread_arg = rows_per_group / thread_count;
  unsigned extra_rows_arg = rows_per_group % thread_count;

  // kernel arguments
  void *kernel_args[] = {&x_gpu_arg,
                         &A_data_gpu,
                         &sq_norms_gpu,
                         &b_gpu,
                         (void *)&iterations,
                         (void *)&bandwidth,
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
}
