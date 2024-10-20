#ifndef SPARSE_HPP
#define SPARSE_HPP

#include "types.hpp"

class SparseLinearSystem {
private:
  SparseMatrix A_;
  Vector b_;
public:
  const SparseMatrix& A() const {
    return this->A_;
  }

  const Vector& b() const {
    return this->b_;
  }

  unsigned row_count() const {
    return this->A_.rows();
  }

  unsigned column_count() const {
    return this->A_.cols();
  }
};

#endif // SPARSE_HPP
