#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <random>

#include "types.hpp"

class SparseLinearSystem {
private:
  SparseMatrix _A;
  Vector _b;

  SparseLinearSystem(const SparseMatrix &A, const Vector &b) : _A(A), _b(b) {}

public:
  const SparseMatrix &A() const { return this->_A; }

  const Vector &b() const { return this->_b; }

  unsigned row_count() const { return this->_A.rows(); }

  unsigned column_count() const { return this->_A.cols(); }

  Vector eigen_solve() const;

  static SparseLinearSystem generate_random_banded_regular(std::mt19937 &rng,
                                                           unsigned dim,
                                                           unsigned bandwidth);

  static SparseLinearSystem read_from_file(std::string path);
};

#endif // SPARSE_HPP
