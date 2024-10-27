#include "dense.hpp"

#include <Eigen/Dense>
#include <random>

#include "types.hpp"

Vector DenseLinearSystem::eigen_solve() const {
  Eigen::Map<const DenseMatrix> A_m(this->A(), this->_row_count,
                                    this->_column_count);
  Eigen::Map<const Vector> b_m(this->b(), this->_row_count);
  return A_m.fullPivLu().solve(b_m);
}

DenseLinearSystem DenseLinearSystem::generate_random_regular(
    std::mt19937& rng, const unsigned dim) {
  DenseLinearSystem lse(dim, dim);

  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto generate_element = [&dist, &rng]() { return dist(rng); };

  Eigen::Map<DenseMatrix> A_m(&lse._A[0], dim, dim);

  do {
    std::generate(lse._A.begin(), lse._A.end(), generate_element);
  } while (A_m.fullPivLu().rank() !=
           dim);  // check if matrix is full-rank (will be full-rank practically
                  // always, but still)

  std::generate(lse._b.begin(), lse._b.end(), generate_element);

  return lse;
}
