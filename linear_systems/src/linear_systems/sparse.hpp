#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <cassert>
#include <random>

#include "types.hpp"

class SparseLinearSystem {
 private:
  SparseMatrix _A;
  Vector _b;

 public:
  SparseLinearSystem(const SparseMatrix &A, const Vector &b) : _A(A), _b(b) {
    assert(A.rows() == b.size() &&
           "Number of rows in coefficient matrix must equal RHS vector "
           "dimension!");
  }

  const SparseMatrix &A() const { return this->_A; }

  const Vector &b() const { return this->_b; }

  unsigned row_count() const { return this->_A.rows(); }

  unsigned column_count() const { return this->_A.cols(); }

  Vector eigen_solve() const;

  static SparseLinearSystem generate_random_banded_regular(std::mt19937 &rng,
                                                           unsigned dim,
                                                           unsigned bandwidth);

  static SparseLinearSystem read_from_stream(std::istream &in_stream);
  void write_to_stream(std::ostream &out_stream) const;
};

#endif  // SPARSE_HPP
