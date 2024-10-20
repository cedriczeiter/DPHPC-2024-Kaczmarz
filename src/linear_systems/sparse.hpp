#ifndef SPARSE_HPP
#define SPARSE_HPP

#include "types.hpp"

class SparseLinearSystem {
private:
  SparseMatrix A_;
  Eigen::VectorXd b_;
public:
  const SparseMatrix& A() const;
  const Vector& b() const;
  unsigned row_count() const;
  unsigned column_count() const;
};

#endif // SPARSE_HPP
