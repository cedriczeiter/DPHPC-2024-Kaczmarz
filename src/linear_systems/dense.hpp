#ifndef DENSE_HPP
#define DENSE_HPP

#include <random>
#include <vector>

#include "types.hpp"

class DenseLinearSystem {
private:
  std::vector<double> _A;
  std::vector<double> _b;
  unsigned _row_count;
  unsigned _column_count;
  
  DenseLinearSystem(unsigned row_count, unsigned column_count) {
    _A = std::vector<double>(row_count * column_count);
    _b = std::vector<double>(row_count);
    _row_count = row_count;
    _column_count = column_count;
  }
public:
  const double *A() const {
    return &_A[0];
  }

  const double *b() const {
    return &_b[0];
  }

  unsigned row_count() const {
    return _row_count;
  }

  unsigned column_count() const {
    return _column_count;
  }

  Vector eigen_solve() const;

  static DenseLinearSystem generate_random_regular(std::mt19937& rng, unsigned dim);
};

#endif // DENSE_HPP
