#include "sparse.hpp"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseQR>
#include <iostream>
#include <random>
#include <vector>

Vector SparseLinearSystem::eigen_solve() const {
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.compute(this->_A);
  return qr.solve(this->_b);
}

Vector SparseLinearSystem::eigen_BiCGSTAB() const {
  Eigen::BiCGSTAB<SparseMatrix,  Eigen::IncompleteLUT<double>> solver;
  solver.compute(this->_A);
  return solver.solve(this->_b);
}

Vector SparseLinearSystem::eigen_CG() const {
    Eigen::ConjugateGradient<SparseMatrix, Eigen::Lower | Eigen::Upper> solver;
    solver.compute(this->_A);
  return solver.solve(this->_b);
}

/**
 * Precondition: A is compressed
 */
bool is_regular(const SparseMatrix& A) {
  if (A.rows() != A.cols()) {
    return false;
  }
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.compute(A);
  return A.rows() == qr.rank();
}

SparseLinearSystem SparseLinearSystem::generate_random_banded_regular(
    std::mt19937& rng, const unsigned dim, const unsigned bandwidth) {
  return BandedLinearSystem::generate_random_regular(rng, dim, bandwidth)
      .to_sparse_system();
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

SparseMatrix sparse_matrix_from_banded(const std::vector<double>& data,
                                       const unsigned dim,
                                       const unsigned bandwidth) {
  SparseMatrix A(dim, dim);
  unsigned idx = 0;
  for (int i = 0; i < (int)dim; i++) {
    const int from_col = std::max(i - (int)bandwidth, 0);
    const int to_col = std::min(i + (int)bandwidth, (int)dim - 1);
    for (int j = from_col; j <= to_col; j++) {
      A.insert(i, j) = data[idx++];
    }
  }
  A.makeCompressed();
  return A;
}

SparseLinearSystem BandedLinearSystem::to_sparse_system() const {
  SparseMatrix A(this->_dim, this->_dim);
  unsigned idx = 0;
  for (int i = 0; i < (int)this->_dim; i++) {
    const int from_col = std::max(i - (int)this->_bandwidth, 0);
    const int to_col = std::min(i + (int)this->_bandwidth, (int)this->_dim - 1);
    for (int j = from_col; j <= to_col; j++) {
      A.insert(i, j) = this->_A_data[idx++];
    }
  }
  return SparseLinearSystem(A, this->_b);
}

BandedLinearSystem BandedLinearSystem::generate_random_regular(
    std::mt19937& rng, const unsigned dim, const unsigned bandwidth) {
  assert(dim >= 2 * bandwidth);
  const unsigned element_count =
      dim * (2 * bandwidth + 1) - bandwidth * (bandwidth + 1);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> data(element_count);
  do {
    std::generate(data.begin(), data.end(),
                  [&dist, &rng] { return dist(rng); });
  } while (!is_regular(sparse_matrix_from_banded(data, dim, bandwidth)));
  const Vector b =
      Vector::NullaryExpr(dim, [&rng, &dist] { return dist(rng); });
  return BandedLinearSystem(dim, bandwidth, data, b);
}
