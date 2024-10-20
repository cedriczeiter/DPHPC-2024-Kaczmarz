#include "sparse.hpp"

const SparseMatrix& SparseLinearSystem::A() const {
  return this->A_;
}

const Vector& SparseLinearSystem::b() const {
  return this->b_;
}

unsigned SparseLinearSystem::row_count() const {
  return this->A_.rows();
}

unsigned SparseLinearSystem::column_count() const {
  return this->A_.cols();
}
