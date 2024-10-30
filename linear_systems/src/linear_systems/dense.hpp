#ifndef DENSE_HPP
#define DENSE_HPP

#include <random>
#include <vector>

#include "types.hpp"

/**
 * @class DenseLinearSystem
 * @brief Represents a linear system with a dense matrix, providing methods for
 *        solving and generating random regular systems.
 */
class DenseLinearSystem {
 private:
  /**
   * @brief The stiffness matrix stored in row-major order.
   */
  std::vector<double> _A;
  /**
   * @brief The load vector.
   */
  std::vector<double> _b;

  unsigned _row_count;
  unsigned _column_count;

  /**
   * @brief Constructs an empty dense linear system with specified dimensions.
   * @param row_count Number of rows in the system matrix and nr elements in
   * vector.
   * @param column_count Number of columns in the system matrix.
   */
  DenseLinearSystem(const unsigned row_count, const unsigned column_count) {
    _A = std::vector<double>(row_count * column_count);
    _b = std::vector<double>(row_count);
    _row_count = row_count;
    _column_count = column_count;
  }

 public:
  /**
   * @brief Accessor for the stiffness matrix.
   * @return Pointer to the start of the stiffness matrix in row-major order.
   */
  const double *A() const { return &this->_A[0]; }

  /**
   * @brief Accessor for the load vector.
   * @return Pointer to the start of the load vector.
   */
  const double *b() const { return &this->_b[0]; }

  /**
   * @brief Returns the row count of the system matrix.
   * @return Number of rows in the system matrix as unsigned int.
   */
  unsigned row_count() const { return this->_row_count; }

  /**
   * @brief Returns the column count of the system matrix.
   * @return Number of columns in the system matrix as unsigned int.
   */
  unsigned column_count() const { return this->_column_count; }

  /**
   * @brief Solves the linear system using full LU decomposition with pivoting
   *        and returns the solution.
   * @return Solution as Eigen::VectorXd computed using Eigen's LU
   * decomposition.
   */
  Vector eigen_solve() const;

  /**
   * @brief Generates a random dense linear system with a full-rank matrix.
   * @param rng Random number generator.
   * @param dim Dimension of the square matrix to generate.
   * @return A random DenseLinearSystem object with dimension `dim` x `dim` and
   * a full-rank matrix.
   */
  static DenseLinearSystem generate_random_regular(std::mt19937 &rng,
                                                   unsigned dim);
};

#endif  // DENSE_HPP
