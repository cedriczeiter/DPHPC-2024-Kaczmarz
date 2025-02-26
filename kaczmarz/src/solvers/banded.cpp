#include "banded.hpp"

#include <cassert>
#include <numeric>

#include "banded_cuda.hpp"
#include "omp.h"

/**
 * Converts a `BandedLinearSystem` and an initial guess for `x` to an
 * `UnpackedBandedSystem`. Pads the resulting LSE to have dimension
 * `target_dim`.
 */
UnpackedBandedSystem unpack_banded_system(const BandedLinearSystem& lse,
                                          const Vector& x,
                                          const unsigned target_dim) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  const std::vector<double> x_padded = [target_dim, bandwidth, &x]() {
    std::vector<double> v(bandwidth + target_dim + bandwidth, 0.0);
    std::copy(x.begin(), x.end(), v.begin() + bandwidth);
    return v;
  }();

  const std::vector<double> A_data = [dim, target_dim, bandwidth, &lse]() {
    std::vector<double> v(target_dim * (2 * bandwidth + 1), 0.0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
      const unsigned to_copy_count = row_idx + 1 + bandwidth;
      std::copy_n(
          lse.A_data().begin() + elem_idx, to_copy_count,
          v.begin() + row_idx * (2 * bandwidth + 1) + (bandwidth - row_idx));
      elem_idx += to_copy_count;
    }
    const unsigned middle_to_copy_count =
        (dim - 2 * bandwidth) * (2 * bandwidth + 1);
    std::copy_n(lse.A_data().begin() + elem_idx, middle_to_copy_count,
                v.begin() + bandwidth * (2 * bandwidth + 1));
    elem_idx += middle_to_copy_count;
    for (unsigned row_i = 0; row_i < bandwidth; row_i++) {
      const unsigned to_copy_count = 2 * bandwidth - row_i;
      std::copy_n(lse.A_data().begin() + elem_idx, to_copy_count,
                  v.begin() + (dim - bandwidth + row_i) * (2 * bandwidth + 1));
      elem_idx += to_copy_count;
    }
    for (unsigned pad_row_idx = dim; pad_row_idx < target_dim; pad_row_idx++) {
      v[pad_row_idx * (2 * bandwidth + 1) + bandwidth] = 1.0;
    }
    return v;
  }();

  const std::vector<double> sq_norms = [bandwidth, dim, target_dim, &lse]() {
    std::vector<double> v(target_dim, 1.0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const unsigned row_nnz =
          std::min({2 * bandwidth + 1, bandwidth + 1 + row_idx,
                    bandwidth + 1 + (dim - 1 - row_idx)});
      v[row_idx] = std::inner_product(lse.A_data().begin() + elem_idx,
                                      lse.A_data().begin() + elem_idx + row_nnz,
                                      lse.A_data().begin() + elem_idx, 0.0);
      elem_idx += row_nnz;
    }
    return v;
  }();

  const std::vector<double> b = [target_dim, &lse]() {
    std::vector<double> v(target_dim, 0.0);
    std::copy(lse.b().begin(), lse.b().end(), v.begin());
    return v;
  }();

  return {bandwidth, target_dim, A_data, x_padded, sq_norms, b};
}

void write_back_solution(const UnpackedBandedSystem& sys, Vector& x) {
  std::copy_n(sys.x_padded.begin() + sys.bandwidth, x.size(), x.begin());
}

void BandedSolver::run_iterations(const BandedLinearSystem& lse, Vector& x,
                                  const unsigned iterations) {
  UnpackedBandedSystem sys = unpack_banded_system(
      lse, x, this->pad_dimension(lse.dim(), lse.bandwidth()));
  this->setup(&sys);
  this->iterate(iterations);
  this->flush_x();
  write_back_solution(sys, x);
  this->cleanup();
}

KaczmarzSolverStatus BandedSolver::solve(const BandedLinearSystem& lse,
                                         Vector& x,
                                         const unsigned iterations_step,
                                         const unsigned max_iterations,
                                         const double abs_tolerance) {
  UnpackedBandedSystem sys = unpack_banded_system(
      lse, x, this->pad_dimension(lse.dim(), lse.bandwidth()));
  const SparseLinearSystem sparse_lse = lse.to_sparse_system();
  this->setup(&sys);
  unsigned iter = 0;
  while (iter < max_iterations) {
    this->iterate(iterations_step);
    iter += iterations_step;
    this->flush_x();
    write_back_solution(sys, x);
    const Vector r = sparse_lse.b() - sparse_lse.A() * x;
    if (r.lpNorm<2>() < abs_tolerance) {
      this->cleanup();
      return KaczmarzSolverStatus::Converged;
    }
  }
  this->cleanup();
  return KaczmarzSolverStatus::OutOfIterations;
}

inline void row_update(UnpackedBandedSystem& sys, const unsigned row_idx) {
  const auto x_iter =
      sys.x_padded.begin() + sys.bandwidth + row_idx - sys.bandwidth;
  const auto row_iter = sys.A_data.begin() + (2 * sys.bandwidth + 1) * row_idx;
  const double dot = std::inner_product(
      row_iter, row_iter + 2 * sys.bandwidth + 1, x_iter, 0.0);
  const double update_coeff = (sys.b[row_idx] - dot) / sys.sq_norms[row_idx];
  std::transform(x_iter, x_iter + 2 * sys.bandwidth + 1, row_iter, x_iter,
                 [update_coeff](const double xi, const double ai) {
                   return xi + update_coeff * ai;
                 });
}

void CPUBandedSolver::setup(UnpackedBandedSystem* const sys) {
  this->sys = sys;
}

void CPUBandedSolver::cleanup() { this->sys = nullptr; }

unsigned OpenMPGrouping1BandedSolver::pad_dimension(const unsigned dim,
                                                    const unsigned bandwidth) {
  const unsigned rows_per_group =
      std::max(2 * bandwidth, ceil_div(dim, 2 * this->thread_count));
  return rows_per_group * 2 * this->thread_count;
}

void OpenMPGrouping1BandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;

  assert(dim % (2 * thread_count) == 0);
  const unsigned rows_per_group = dim / (2 * thread_count);

#pragma omp parallel num_threads(this->thread_count)
  {
    const unsigned tid = omp_get_thread_num();
    const unsigned base_row_idx1 = rows_per_group * (2 * tid);
    const unsigned base_row_idx2 = rows_per_group * (2 * tid + 1);
    for (unsigned iter = 0; iter < iterations; iter++) {
      for (unsigned row_idx = base_row_idx1;
           row_idx < base_row_idx1 + rows_per_group; row_idx++) {
        row_update(*this->sys, row_idx);
      }

#pragma omp barrier

      for (unsigned row_idx = base_row_idx2;
           row_idx < base_row_idx2 + rows_per_group; row_idx++) {
        row_update(*this->sys, row_idx);
      }

#pragma omp barrier
    }
  }
}

unsigned OpenMPGrouping2BandedSolver::pad_dimension(const unsigned dim,
                                                    const unsigned bandwidth) {
  const unsigned group_count = 2 * bandwidth + 1;
  const unsigned rows_per_group = ceil_div(dim, group_count);
  return rows_per_group * group_count;
}

void OpenMPGrouping2BandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;
  const unsigned bandwidth = sys->bandwidth;

  assert(dim % (2 * bandwidth + 1) == 0);
  const unsigned rows_per_group = dim / (2 * bandwidth + 1);

#pragma omp parallel num_threads(this->thread_count)
  {
    const unsigned tid = omp_get_thread_num();
    const unsigned rows_per_thread = rows_per_group / thread_count;
    const unsigned extra_rows = rows_per_group % thread_count;
    const unsigned row_in_group_from =
        tid * rows_per_thread + std::min(tid, extra_rows);
    const unsigned row_in_group_to =
        (tid + 1) * rows_per_thread + std::min(tid + 1, extra_rows);
    for (unsigned iter = 0; iter < iterations; iter++) {
      for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1; group_idx++) {
        for (unsigned row_in_group_idx = row_in_group_from;
             row_in_group_idx < row_in_group_to; row_in_group_idx++) {
          const unsigned row_idx =
              row_in_group_idx * (2 * bandwidth + 1) + group_idx;
          row_update(*this->sys, row_idx);
        }
#pragma omp barrier
      }
    }
  }
}

unsigned SerialNaiveBandedSolver::pad_dimension(
    const unsigned dim, const unsigned /* bandwidth */) {
  return dim;
}

void SerialNaiveBandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;

  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      row_update(*this->sys, row_idx);
    }
  }
}

unsigned SerialInterleavedBandedSolver::pad_dimension(
    const unsigned dim, const unsigned /* bandwidth */) {
  return dim;
}

void SerialInterleavedBandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;
  const unsigned bandwidth = sys->bandwidth;

  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1; group_idx++) {
      for (unsigned row_idx = group_idx; row_idx < dim;
           row_idx += 2 * bandwidth + 1) {
        row_update(*this->sys, row_idx);
      }
    }
  }
}

template <typename T>
static T* gpu_malloc_and_copy(const std::vector<T>& v) {
  const size_t byte_count = v.size() * sizeof(T);
  T* const gpu_memory = (T*)cuda_malloc(byte_count);
  cuda_memcpy_host_to_device(gpu_memory, &v[0], byte_count);
  return gpu_memory;
}

void GPUBandedSolver::setup(UnpackedBandedSystem* const sys) {
  this->cleanup();
  this->sys = sys;
  this->x_gpu = gpu_malloc_and_copy(sys->x_padded);
  this->A_data_gpu = gpu_malloc_and_copy(sys->A_data);
  this->sq_norms_gpu = gpu_malloc_and_copy(sys->sq_norms);
  this->b_gpu = gpu_malloc_and_copy(sys->b);
}

void GPUBandedSolver::flush_x() {
  const size_t byte_count = this->sys->x_padded.size() * sizeof(double);
  cuda_memcpy_device_to_host(&this->sys->x_padded[0], x_gpu, byte_count);
}

void GPUBandedSolver::cleanup() {
  if (this->x_gpu) {
    cuda_free(this->x_gpu);
    this->x_gpu = nullptr;
  }
  if (this->A_data_gpu) {
    cuda_free(this->A_data_gpu);
    this->A_data_gpu = nullptr;
  }
  if (this->sq_norms_gpu) {
    cuda_free(this->sq_norms_gpu);
    this->sq_norms_gpu = nullptr;
  }
  if (this->b_gpu) {
    cuda_free(this->b_gpu);
    this->b_gpu = nullptr;
  }
  this->sys = nullptr;
}

unsigned CUDAGrouping1BandedSolver::pad_dimension(const unsigned dim,
                                                  const unsigned bandwidth) {
  const unsigned thread_count = block_count * threads_per_block;
  return std::max(dim, bandwidth * 2 * 2 * thread_count);
}

void CUDAGrouping1BandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = this->sys->dim;
  const unsigned bandwidth = this->sys->bandwidth;

  const unsigned group_count = 2 * block_count * threads_per_block;

  invoke_banded_grouping1_kernel(
      block_count, threads_per_block, this->x_gpu + bandwidth, this->A_data_gpu,
      this->sq_norms_gpu, this->b_gpu, iterations, bandwidth, dim / group_count,
      dim % group_count);
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

  assert(dim % (2 * bandwidth + 1) == 0);

  const unsigned rows_per_group = dim / (2 * bandwidth + 1);

  const unsigned thread_count = block_count * threads_per_block;

  invoke_banded_grouping2_kernel(
      block_count, threads_per_block, this->x_gpu + bandwidth, this->A_data_gpu,
      this->sq_norms_gpu, this->b_gpu, iterations, bandwidth,
      rows_per_group / thread_count, rows_per_group % thread_count);
}
