#include "kaczmarz_banded.hpp"

#include <cassert>
#include <iostream>
#include <numeric>

#include "omp.h"

KaczmarzSolverStatus kaczmarz_banded_2_cpu_threads(
    const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations,
    double precision) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  assert(2 * bandwidth <= dim);

  const std::vector<double> sq_norms = [bandwidth, dim, &lse]() {
    std::vector<double> sq_norms(dim, 0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const unsigned row_nnz =
          std::min({2 * bandwidth + 1, bandwidth + 1 + row_idx,
                    bandwidth + 1 + (dim - 1 - row_idx)});
      sq_norms[row_idx] =
          std::inner_product(lse.A_data().begin() + elem_idx,
                             lse.A_data().begin() + elem_idx + row_nnz,
                             lse.A_data().begin() + elem_idx, 0.0);
      elem_idx += row_nnz;
    }
    return sq_norms;
  }();

  const auto row_update = [precision, &lse, &x, &sq_norms](
                              const unsigned row_idx, const unsigned elem_idx,
                              const unsigned x_base_idx,
                              const unsigned row_nnz) -> bool {
    const auto x_iter = x.begin() + x_base_idx;
    const auto row_iter = lse.A_data().begin() + elem_idx;
    const double dot =
        std::inner_product(row_iter, row_iter + row_nnz, x_iter, 0.0);
    const double update_coeff = (lse.b()[row_idx] - dot) / sq_norms[row_idx];
    std::transform(x_iter, x_iter + row_nnz, row_iter, x_iter,
                   [update_coeff](const double xi, const double ai) {
                     return xi + update_coeff * ai;
                   });
    return precision <= fabs(update_coeff);
  };

  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_update = false;

    // update rows at the very top
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + row_idx;
      if (row_update(row_idx, elem_idx, 0, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }

    const unsigned middle_row_count = (dim - 2 * bandwidth) / 4 * 4;

    assert(middle_row_count / 4 >= 2 * bandwidth);

    const auto full_row_update = [bandwidth, elem_idx,
                                  &row_update](const unsigned row_idx) -> bool {
      return row_update(row_idx,
                        elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1),
                        row_idx - bandwidth, 2 * bandwidth + 1);
    };

#pragma omp parallel num_threads(2)
    {
      const int id = omp_get_thread_num();
      const unsigned start_row_idx = bandwidth + middle_row_count / 2 * id;
      const unsigned batch_row_count = middle_row_count / 4;
      for (unsigned row_idx = start_row_idx;
           row_idx < start_row_idx + batch_row_count; row_idx++) {
        if (full_row_update(row_idx)) {
#pragma omp atomic write
          substantial_update = true;
        }
      }
    }

#pragma omp parallel num_threads(2)
    {
      const int id = omp_get_thread_num();
      const unsigned start_row_idx =
          bandwidth + middle_row_count / 4 + middle_row_count / 2 * id;
      const unsigned batch_row_count = middle_row_count / 4;
      for (unsigned row_idx = start_row_idx;
           row_idx < start_row_idx + batch_row_count; row_idx++) {
        if (full_row_update(row_idx)) {
// TODO: this algorithm should be deterministic, but it seems it isn't? why?
#pragma omp atomic write
          substantial_update = true;
        }
      }
    }

    for (unsigned row_idx = bandwidth + middle_row_count;
         row_idx < dim - bandwidth; row_idx++) {
      if (row_update(row_idx,
                     elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1),
                     row_idx - bandwidth, 2 * bandwidth + 1)) {
        substantial_update = true;
      }
    }

    // update rows at the very bottom
    elem_idx += (dim - 2 * bandwidth) * (2 * bandwidth + 1);
    for (unsigned row_idx = dim - bandwidth; row_idx < dim; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + (dim - 1 - row_idx);
      if (row_update(row_idx, elem_idx, dim - row_nnz, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }
    if (!substantial_update) {
      std::cout << iter << " iterations" << std::endl;
      return KaczmarzSolverStatus::Converged;
    }
  }

  return KaczmarzSolverStatus::OutOfIterations;
}

KaczmarzSolverStatus kaczmarz_banded_serial(const BandedLinearSystem& lse,
                                            Eigen::VectorXd& x,
                                            unsigned max_iterations,
                                            double precision) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  assert(2 * bandwidth <= dim);

  const std::vector<double> sq_norms = [bandwidth, dim, &lse]() {
    std::vector<double> sq_norms(dim, 0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const unsigned row_nnz =
          std::min({2 * bandwidth + 1, bandwidth + 1 + row_idx,
                    bandwidth + 1 + (dim - 1 - row_idx)});
      sq_norms[row_idx] =
          std::inner_product(lse.A_data().begin() + elem_idx,
                             lse.A_data().begin() + elem_idx + row_nnz,
                             lse.A_data().begin() + elem_idx, 0.0);
      elem_idx += row_nnz;
    }
    return sq_norms;
  }();

  const auto row_update = [precision, &lse, &x, &sq_norms](
                              const unsigned row_idx, const unsigned elem_idx,
                              const unsigned x_base_idx,
                              const unsigned row_nnz) -> bool {
    const auto x_iter = x.begin() + x_base_idx;
    const auto row_iter = lse.A_data().begin() + elem_idx;
    const double dot =
        std::inner_product(row_iter, row_iter + row_nnz, x_iter, 0.0);
    const double update_coeff = (lse.b()[row_idx] - dot) / sq_norms[row_idx];
    std::transform(x_iter, x_iter + row_nnz, row_iter, x_iter,
                   [update_coeff](const double xi, const double ai) {
                     return xi + update_coeff * ai;
                   });
    return precision <= fabs(update_coeff);
  };

  for (unsigned iter = 0; iter < max_iterations; iter++) {
    bool substantial_update = false;

    // update rows at the very top
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + row_idx;
      if (row_update(row_idx, elem_idx, 0, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }

    const auto full_row_update = [bandwidth, elem_idx,
                                  &row_update](const unsigned row_idx) -> bool {
      return row_update(row_idx,
                        elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1),
                        row_idx - bandwidth, 2 * bandwidth + 1);
    };

    for (unsigned row_idx = bandwidth; row_idx < dim - bandwidth; row_idx++) {
      if (full_row_update(row_idx)) {
        substantial_update = true;
      }
    }

    // update rows at the very bottom
    elem_idx += (dim - 2 * bandwidth) * (2 * bandwidth + 1);
    for (unsigned row_idx = dim - bandwidth; row_idx < dim; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + (dim - 1 - row_idx);
      if (row_update(row_idx, elem_idx, dim - row_nnz, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }
    if (!substantial_update) {
      std::cout << iter << " iterations" << std::endl;
      return KaczmarzSolverStatus::Converged;
    }
  }

  return KaczmarzSolverStatus::OutOfIterations;
}
