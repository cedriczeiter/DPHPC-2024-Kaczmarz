#include "sparse.hpp"

#include <Eigen/SparseQR>
#include <iostream>
#include <random>
#include <vector>

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
    qr.setPivotThreshold(1e-6);
    qr.compute(A);
    rank = qr.rank();
  } while (rank != (int)dim);
  const Vector b =
      Vector::NullaryExpr(dim, [&rng, &dist] { return dist(rng); });
  return SparseLinearSystem(A, b);
}

SparseLinearSystem SparseLinearSystem::read_from_stream(
    std::istream& in_stream) {
  unsigned nnz, rows, cols;
  in_stream >> nnz >> rows >> cols;

  assert(nnz <= rows * cols);

  std::vector<Eigen::Triplet<double>> triplets_A;
  triplets_A.reserve(nnz);

  // every next three entries correspond to values for a triplet
  for (unsigned i = 0; i < nnz; i++) {
    unsigned row, col;
    double value;
    in_stream >> row >> col >> value;
    triplets_A.emplace_back(row, col, value);
  }

  SparseMatrix A(rows, cols);
  A.setFromTriplets(triplets_A.begin(), triplets_A.end());

  // construct rhs vector
  Vector b = Vector::Zero(rows);
  for (unsigned i = 0; i < cols; i++) {
    in_stream >> b[i];
  }

  return SparseLinearSystem(A, b);
}

// idea for export Linear System:
// 1st entry: nnz in Matrix, 2nd and 3rd entry: rows/cols of Matrix, then
// triplets printed out, then values of RHS Vector
void SparseLinearSystem::write_to_stream(std::ostream& out_stream) const {
  out_stream << this->_A.nonZeros() << '\n';
  out_stream << this->_A.rows() << " " << this->_A.cols() << '\n';

  // write values of matrix
  for (int k = 0; k < this->_A.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(this->_A, k); it; ++it) {
      out_stream << it.row() << " " << it.col() << " " << it.value() << '\n';
    }
  }

  // write values of vector
  for (int i = 0; i < this->_b.size(); i++) out_stream << this->_b[i] << '\n';

  out_stream.flush();
}
