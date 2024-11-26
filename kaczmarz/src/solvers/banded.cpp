#include "banded.hpp"

#include <cassert>
#include <numeric>

#include "banded_cuda.hpp"
#include "omp.h"

KaczmarzSolverStatus kaczmarz_banded_2_cpu_threads(
    const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations,
    double precision) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  // Necessary for the separate processing of the first and last rows to work
  // out.
  assert(2 * bandwidth <= dim);

  // Necessary for the division and parallel processing of the middle rows to
  // work out. Otherwise, the parts of the result vector x that the
  // parallel-running threads write to might overlap. -> Race conditions.
  // Specifically, say in the first part, both threads access a subarray of
  // length `middle_row_count / 4 + (2 * bandwidth + 1)`. Thread 0 starts it at
  // idx. 0 while thread 1 at middle_row_count / 2. That means that we need
  // `middle_row_count / 4 + (2 * bandwidth + 1) <= middle_row_count / 2` i.e.
  // `2 * bandwidth + 1 <= middle_row_count / 4` i.e.
  // `8 * bandwidth + 4 <= (dim - 2 * bandwidth)` i.e.
  // `10 * bandwidth + 4 <= dim`
  assert(10 * bandwidth + 4 <= dim);

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
    /*
       We can group the rows of the coefficient matrix into groups A, B, C, D,
       E, F, G as
       A
       B
       C
       D
       E
       F
       G
       with row 0 being at the top and row dim - 1 at the bottom.
       There are `bandwidth` rows in each of groups A and G. In those rows, the
       number of potentially non-zero entries is less than the full (2 *
       bandwidth + 1) which it is in all other rows.

       For convenience, we want to divide the remaining `dim - 2 * bandwidth`
       rows into four equally-sized groups. Those are B, C, D, and E, while F is
       between 0 and 3 rows that needed to be taken away to make the number of
       rows divisible by 4.

       Then if the dimension is relatively large compared to the bandwidth, this
       division allows us to process rows in group B in parallel with rows in
       group D because any row in B is orthogonal to any row in D. So we do that
       and finally repeat the same with groups C and E.
     */

    bool substantial_update = false;

    // update rows at the very top (group A)
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + row_idx;
      if (row_update(row_idx, elem_idx, 0, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }

    const unsigned middle_row_count = (dim - 2 * bandwidth) / 4 * 4;

    assert(middle_row_count / 4 >= 2 * bandwidth + 1);

    const auto full_row_update = [bandwidth, elem_idx,
                                  &row_update](const unsigned row_idx) -> bool {
      return row_update(row_idx,
                        elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1),
                        row_idx - bandwidth, 2 * bandwidth + 1);
    };

#pragma omp parallel num_threads(2)
    {
      // thread 0 processes group B and thread 1 group D

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
      // thread 0 processes group C and thread 1 group E

      const int id = omp_get_thread_num();
      const unsigned start_row_idx =
          bandwidth + middle_row_count / 4 + middle_row_count / 2 * id;
      const unsigned batch_row_count = middle_row_count / 4;
      for (unsigned row_idx = start_row_idx;
           row_idx < start_row_idx + batch_row_count; row_idx++) {
        if (full_row_update(row_idx)) {
#pragma omp atomic write
          substantial_update = true;
        }
      }
    }

    // process group F
    for (unsigned row_idx = bandwidth + middle_row_count;
         row_idx < dim - bandwidth; row_idx++) {
      if (row_update(row_idx,
                     elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1),
                     row_idx - bandwidth, 2 * bandwidth + 1)) {
        substantial_update = true;
      }
    }

    // update rows at the very bottom (group G)
    elem_idx += (dim - 2 * bandwidth) * (2 * bandwidth + 1);
    for (unsigned row_idx = dim - bandwidth; row_idx < dim; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + (dim - 1 - row_idx);
      if (row_update(row_idx, elem_idx, dim - row_nnz, row_nnz)) {
        substantial_update = true;
      }
      elem_idx += row_nnz;
    }
    if (!substantial_update) {
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
    // Same idea as in the implementation `kaczmarz_banded_2_cpu_threads`.
    // Except that without parallelization, we can merge groups B, C, D, E, and
    // F all together.

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
      return KaczmarzSolverStatus::Converged;
    }
  }

  return KaczmarzSolverStatus::OutOfIterations;
}

KaczmarzSolverStatus kaczmarz_banded_cuda(const BandedLinearSystem& lse,
                                          Eigen::VectorXd& x,
                                          const unsigned /* max_iterations */,
                                          const double /* precision */) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  // reshuffling / padding memory on the CPU

  const unsigned thread_count = (dim - 1) / (2 * bandwidth + 1) + 1;
  const unsigned dim_padded = thread_count * (2 * bandwidth + 1);
  std::vector<double> x_padded(bandwidth + dim_padded + bandwidth, 0.0);
  std::copy(x.begin(), x.end(), x_padded.begin() + bandwidth);
  std::vector<double> A_data_padded(dim_padded * (2 * bandwidth + 1), 0.0);
  unsigned elem_idx = 0;
  for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
    const unsigned to_copy_count = row_idx + 1 + bandwidth;
    std::copy_n(lse.A_data().begin() + elem_idx, to_copy_count,
                A_data_padded.begin() + row_idx * (2 * bandwidth + 1) +
                    (bandwidth - row_idx));
    elem_idx += to_copy_count;
  }
  const unsigned middle_to_copy_count =
      (dim - 2 * bandwidth) * (2 * bandwidth + 1);
  std::copy_n(lse.A_data().begin() + elem_idx, middle_to_copy_count,
              A_data_padded.begin() + bandwidth * (2 * bandwidth + 1));
  elem_idx += middle_to_copy_count;
  for (unsigned row_i = 0; row_i < bandwidth; row_i++) {
    const unsigned to_copy_count = 2 * bandwidth - row_i;
    std::copy_n(lse.A_data().begin() + elem_idx, to_copy_count,
                A_data_padded.begin() +
                    (dim - bandwidth + row_i) * (2 * bandwidth + 1));
    elem_idx += to_copy_count;
  }
  for (unsigned pad_row_idx = dim; pad_row_idx < dim_padded; pad_row_idx++) {
    A_data_padded[pad_row_idx * (2 * bandwidth + 1) + bandwidth] = 1.0;
  }
  const std::vector<double> sq_norms_padded = [bandwidth, dim, dim_padded,
                                               &lse]() {
    std::vector<double> sq_norms(dim_padded, 1.0);
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
  std::vector<double> b_padded(dim_padded, 0.0);
  std::copy(lse.b().begin(), lse.b().end(), b_padded.begin());

  invoke_kaczmarz_banded_update(bandwidth, dim_padded, thread_count,
                                A_data_padded, x_padded, sq_norms_padded,
                                b_padded);

  std::copy_n(x_padded.begin() + bandwidth, dim, x.begin());

  return KaczmarzSolverStatus::Converged;
}
