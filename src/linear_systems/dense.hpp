#ifndef DENSE_HPP
#define DENSE_HPP

#include <random>
#include <vector>

#include "types.hpp"

class DenseLinearSystem {
private:
  /**
   * The stiffness matrix in row-major order
   */
  std::vector<double> _A;
  /**
   * The load vector
   */
  std::vector<double> _b;
  unsigned _row_count;
  unsigned _column_count;
  
  DenseLinearSystem(const unsigned row_count, const unsigned column_count) {
    _A = std::vector<double>(row_count * column_count);
    _b = std::vector<double>(row_count);
    _row_count = row_count;
    _column_count = column_count;
  }
public:
  /**
   * The stiffness matrix in row-major order
   */
  const double *A() const {
    return &this->_A[0];
  }

  /**
   * The load vector
   */
  const double *b() const {
    return &this->_b[0];
  }

  unsigned row_count() const {
    return this->_row_count;
  }

  unsigned column_count() const {
    return this->_column_count;
  }

  /**
   * Computes and returns the solution that Eigen computes with full LU pivoting.
   */
  Vector eigen_solve() const;

  /**
   * Generates a random dense LSE with a regular stiffness matrix of dimension dim x dim.
   * Each element of both the stiffness matrix and the load vector is drawn i.i.d. from a uniform distribution.
   */
  static DenseLinearSystem generate_random_regular(std::mt19937& rng, unsigned dim);
};

#endif // DENSE_HPP
