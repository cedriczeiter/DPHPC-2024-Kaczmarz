#include "sparse.hpp"

#include <Eigen/SparseQR>
#include <random>

Vector SparseLinearSystem::eigen_solve() const {
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.compute(this->_A);
  return qr.solve(this->_b);
}

SparseLinearSystem SparseLinearSystem::generate_random_banded_regular(
    std::mt19937& rng, const unsigned dim, const unsigned bandwidth) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  SparseMatrix A(dim, dim);
  int rank;
  do {
    for (int i = 0; i < (int)dim; i++) {
      const int from_col = std::max(i - (int)bandwidth, 0);
      const int to_col = std::min(i + (int)bandwidth, (int)dim - 1);
      for (int j = from_col; j <= to_col; j++) {
        A.insert(i, j) = dist(rng);
      }
    }
    A.makeCompressed();
    Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
    qr.compute(A);
    rank = qr.rank();
  } while (rank != (int)dim);
  const Vector b =
      Vector::NullaryExpr(dim, [&rng, &dist] { return dist(rng); });
  return SparseLinearSystem(A, b);
}
