#include "sparse.hpp"

#include <Eigen/SparseQR>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

Vector SparseLinearSystem::eigen_solve() const {
  Eigen::SparseQR<SparseMatrix, Eigen::COLAMDOrdering<int>> qr;
  qr.compute(this->_A);
  return qr.solve(this->_b);
}

SparseLinearSystem SparseLinearSystem::generate_random_banded_regular(
    std::mt19937 &rng, const unsigned dim, const unsigned bandwidth) {
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

SparseLinearSystem SparseLinearSystem::read_from_file(std::string path) {
  std::ifstream inputFile(path);
  std::vector<double> entries;
  if (!inputFile) {
    std::cerr << "Error opening file. Generating random banded regular instead"
              << std::endl;
    std::mt19937 rng(21);
    return generate_random_banded_regular(rng, 10, 3);
  }

  std::cout << "Reading in Matrix entries from " << path << std::endl;
  double entry;
  while (inputFile >> entry) entries.push_back(entry);
  // first entry: nnz in A
  unsigned nnz = (unsigned)entries.at(0);
  // next two entries of file are rows and cols
  unsigned rows = (unsigned)entries.at(1);
  unsigned cols = (unsigned)entries.at(2);
  // assert format is correct
  assert(3 + 3 * nnz + cols == entries.size());

  SparseMatrix A(rows, cols);
  std::vector<Eigen::Triplet<double>> triplets_A;

  // every next three entries correspond to values for a triplet
  for (unsigned i = 0; i < nnz; i++) {
    unsigned idx = 3 + 3 * i;
    unsigned row = (unsigned)entries.at(idx);
    unsigned col = (unsigned)entries.at(idx + 1);
    double value = (unsigned)entries.at(idx + 2);
    triplets_A.push_back(Eigen::Triplet<double>(row, col, value));
  }
  A.setFromTriplets(triplets_A.begin(), triplets_A.end());
  std::cout << "A constructed" << std::endl;
  // construct rhs vector
  Vector b = Vector::Zero(cols);
  for (unsigned i = 0; i < cols; i++) {
    unsigned idx = 3 + 3 * nnz + i;
    b[i] = entries.at(idx);
  }
  std::cout << "RHS Vector constructed" << std::endl;
  return SparseLinearSystem(A, b);
}
