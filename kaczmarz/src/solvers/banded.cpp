#include "banded.hpp"

#include <cassert>
#include <numeric>

#include "banded_cuda.hpp"
#include "omp.h"
#include "unpacked_banded_system.hpp"

unsigned ceil_div(const unsigned a, const unsigned b) {
  assert(b != 0);
  if (a == 0) {
    return 0;
  }
  return (a - 1) / b + 1;
}

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

void BandedSolver::run_iterations(const BandedLinearSystem& lse, Vector& x, const unsigned iterations) {
  UnpackedBandedSystem sys = unpack_banded_system(lse, x, this->pad_dimension(lse.dim(), lse.bandwidth()));
  this->setup(&sys);
  this->iterate(iterations);
  write_back_solution(sys, x);
  this->cleanup();
}

inline void row_update(UnpackedBandedSystem& sys, const unsigned row_idx) {
  const auto x_iter =
    sys.x_padded.begin() + sys.bandwidth + row_idx - sys.bandwidth;
  const auto row_iter =
    sys.A_data.begin() + (2 * sys.bandwidth + 1) * row_idx;
  const double dot = std::inner_product(
      row_iter, row_iter + 2 * sys.bandwidth + 1, x_iter, 0.0);
  const double update_coeff =
    (sys.b[row_idx] - dot) / sys.sq_norms[row_idx];
  std::transform(x_iter, x_iter + 2 * sys.bandwidth + 1, row_iter, x_iter,
      [update_coeff](const double xi, const double ai) {
      return xi + update_coeff * ai;
      });
}

void CPUBandedSolver::setup(UnpackedBandedSystem *const sys) {
  this->sys = sys;
}

void CPUBandedSolver::cleanup() {
  this->sys = nullptr;
}

unsigned OpenMPGrouping1IBandedSolver::pad_dimension(const unsigned dim, const unsigned bandwidth) {
  const unsigned rows_per_group =
      std::max(2 * bandwidth, ceil_div(dim, 2 * this->thread_count));
  return rows_per_group * 2 * this->thread_count;
}

void OpenMPGrouping1IBandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;
  const unsigned bandwidth = sys->bandwidth;

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

unsigned OpenMPGrouping2IBandedSolver::pad_dimension(const unsigned dim, const unsigned bandwidth) {
  const unsigned group_count = 2 * bandwidth + 1;
  const unsigned rows_per_group = ceil_div(dim, group_count);
  return rows_per_group * group_count;
}

void OpenMPGrouping2IBandedSolver::iterate(const unsigned iterations) {
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
      for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1;
           group_idx++) {
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

unsigned SerialNaiveBandedSolver::pad_dimension(const unsigned dim, const unsigned /* bandwidth */) {
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

unsigned SerialInterleavedBandedSolver::pad_dimension(const unsigned dim, const unsigned /* bandwidth */) {
  return dim;
}

void SerialInterleavedBandedSolver::iterate(const unsigned iterations) {
  const unsigned dim = sys->dim;
  const unsigned bandwidth = sys->bandwidth;

  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1;
         group_idx++) {
      for (unsigned row_idx = group_idx; row_idx < dim;
           row_idx += 2 * bandwidth + 1) {
        row_update(*this->sys, row_idx);
      }
    }
  }
}

void kaczmarz_banded_cuda_grouping1(const BandedLinearSystem& lse,
                                    Eigen::VectorXd& x,
                                    const unsigned iterations,
                                    const unsigned block_count,
                                    const unsigned threads_per_block) {
  const unsigned thread_count = block_count * threads_per_block;
  const unsigned dim_padded =
      std::max(lse.dim(), lse.bandwidth() * 2 * 2 * thread_count);
  UnpackedBandedSystem sys = unpack_banded_system(lse, x, dim_padded);
  invoke_kaczmarz_banded_cuda_grouping1(sys, iterations, block_count,
                                        threads_per_block);
  write_back_solution(sys, x);
}

void kaczmarz_banded_cuda_grouping2(const BandedLinearSystem& lse,
                                    Eigen::VectorXd& x,
                                    const unsigned iterations,
                                    const unsigned block_count,
                                    const unsigned threads_per_block) {
  const unsigned group_count = 2 * lse.bandwidth() + 1;
  const unsigned rows_per_group = ceil_div(lse.dim(), group_count);
  const unsigned dim_padded = rows_per_group * group_count;
  UnpackedBandedSystem sys = unpack_banded_system(lse, x, dim_padded);
  invoke_kaczmarz_banded_cuda_grouping2(sys, iterations, block_count,
                                        threads_per_block);
  write_back_solution(sys, x);
}
