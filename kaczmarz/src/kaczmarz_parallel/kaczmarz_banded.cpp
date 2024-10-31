#include "kaczmarz_banded.hpp"
#include <cassert>
#include <numeric>

KaczmarzSolverStatus banded_sparse_kaczmarz(const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations, double precision) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  assert(2 * bandwidth <= dim);

  const std::vector<double> sq_norms = [bandwidth, dim, &lse]() {
    std::vector<double> sq_norms(dim, 0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const unsigned row_nnz = std::min({ 2 * bandwidth + 1, bandwidth + 1 + row_idx, bandwidth + 1 + (dim - 1 - row_idx) });
      sq_norms[row_idx] = std::inner_product(lse.A_data().begin() + elem_idx, lse.A_data().begin() + elem_idx + row_nnz, lse.A_data().begin() + elem_idx, 0.0);
      elem_idx += row_nnz;
    }
    return sq_norms;
  }();

  const auto row_update = [precision, &lse, &x, &sq_norms](const unsigned row_idx, const unsigned elem_idx, const unsigned x_base_idx, const unsigned row_nnz) -> bool {
      const auto x_iter = x.begin() + x_base_idx;
      const auto row_iter = lse.A_data().begin() + elem_idx;
      const double dot = std::inner_product(row_iter, row_iter + row_nnz, x_iter, 0.0);
      const double update_coeff = (lse.b()[row_idx] - dot) / sq_norms[row_idx];
      std::transform(x_iter, x_iter + row_nnz, row_iter, x_iter, [update_coeff](const double xi, const double ai) { return xi + update_coeff * ai; });
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

    // update rows in the middle (most of the total number of rows);
    // parallelizing as much as possible
    for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1; group_idx++) {

      #pragma omp parallel for
      for (unsigned row_idx = bandwidth + group_idx; row_idx < dim - bandwidth; row_idx += 2 * bandwidth + 1) {
        if (row_update(row_idx, elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1), row_idx - bandwidth, 2 * bandwidth + 1)) {
          //#pragma omp atomic write
          substantial_update = true;
        }
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

