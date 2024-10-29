#include "kaczmarz_banded.hpp"
#include <cassert>

KaczmarzSolverStatus banded_sparse_kaczmarz(const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations, double precision) {
  const unsigned bandwidth = lse.bandwidth();
  const unsigned dim = lse.dim();

  assert(2 * bandwidth <= dim);

  const std::vector<double> sq_norms = [bandwidth, dim, &lse]() {
    std::vector<double> sq_norms(dim, 0);
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const unsigned row_nnz = std::min({ 2 * bandwidth + 1, bandwidth + 1 + row_idx, bandwidth + 1 + (dim - 1 - row_idx) });
      for (unsigned elem_in_row_idx = 0; elem_in_row_idx < row_nnz; elem_in_row_idx++) {
        const double v = lse.A_data()[elem_idx + elem_in_row_idx];
        sq_norms[row_idx] += v * v;
      }
      elem_idx += row_nnz;
    }
    return sq_norms;
  }();

  // TODO add precision-based stopping criterion

  for (unsigned iter = 0; iter < max_iterations; iter++) {
    // update rows at the very top
    unsigned elem_idx = 0;
    for (unsigned row_idx = 0; row_idx < bandwidth; row_idx++) {
      const unsigned row_nnz = bandwidth + 1 + row_idx;
      const double dot = [row_nnz, elem_idx, &lse, &x]() {
        double dot = 0;
        for (unsigned elem_in_row_idx = 0; elem_in_row_idx < row_nnz; elem_in_row_idx++) {
          dot += lse.A_data()[elem_idx + elem_in_row_idx] * x[elem_in_row_idx];
        }
        return dot;
      }();
      const double update_coeff = (lse.b()[row_idx] - dot) / sq_norms[row_idx];
      for (unsigned elem_in_row_idx = 0; elem_in_row_idx < row_nnz; elem_in_row_idx++) {
        x[elem_in_row_idx] += update_coeff * lse.A_data()[elem_idx + elem_in_row_idx];
      }
    }

    for (unsigned group_idx = 0; group_idx < 2 * bandwidth + 1; group_idx++) {

      #pragma omp parallel for
      for (unsigned row_idx = bandwidth + group_idx; row_idx < dim - bandwidth; row_idx += 2 * bandwidth + 1) {
        const unsigned fst_elem_idx = elem_idx + (row_idx - bandwidth) * (2 * bandwidth + 1);
        const double dot = [row_idx, fst_elem_idx, bandwidth, &lse, &x]() {
          double dot = 0;
          for (unsigned elem_in_row_idx = 0; elem_in_row_idx < 2 * bandwidth + 1; elem_in_row_idx++) {
            dot += lse.A_data()[fst_elem_idx + elem_in_row_idx] * x[row_idx - bandwidth + elem_in_row_idx];
          }
          return dot;
        }();
        const double update_coeff = (lse.b()[row_idx] - dot) / sq_norms[row_idx];
        for (unsigned elem_in_row_idx = 0; elem_in_row_idx < 2 * bandwidth + 1; elem_in_row_idx++) {
          x[row_idx - bandwidth + elem_in_row_idx] += update_coeff * lse.A_data()[fst_elem_idx + elem_in_row_idx];
        }
      }
    }

    // TODO update rows at the very bottom (analogously to those at the very top)
  }
}

