#ifndef SPARSE_HPP
#define SPARSE_HPP

#include <cassert>
#include <random>

#include "types.hpp"

/**
 * @class SparseLinearSystem
 * @brief Represents a sparse linear system, providing methods for solving,
 *        generating random banded matrices, and reading/writing from streams.
 */
class SparseLinearSystem {
 private:
  SparseMatrix _A;
  Vector _b;

 public:
  /**
   * @brief Constructs a sparse linear system with given matrix and vector.
   * @param A Sparse matrix representing the system's coefficients of type
   * Eigen::SparseMatrix<double, Eigen::RowMajor>.
   * @param b Vector representing the right-hand side of the system of type
   * Eigen::VectorXd .
   * @note Asserts that the row count of A matches the size of b.
   */
  SparseLinearSystem(const SparseMatrix &A, const Vector &b) : _A(A), _b(b) {
    assert(A.rows() == b.size() &&
           "Number of rows in coefficient matrix must equal RHS vector "
           "dimension!");
  }

  /**
   * @brief Accessor for the sparse coefficient matrix.
   * @return Reference to the sparse matrix representing the system's
   * coefficients of type Eigen::SparseMatrix<double, Eigen::RowMajor>.
   */
  const SparseMatrix &A() const { return this->_A; }

  /**
   * @brief Accessor for the sparse coefficient matrix.
   * @return Reference to the sparse matrix representing the system's
   * coefficients of type Eigen::VectorXd.
   */
  const Vector &b() const { return this->_b; }

  /**
   * @brief Returns the row count of the sparse matrix.
   * @return Number of rows in the coefficient matrix as unsigned int.
   */
  unsigned row_count() const { return this->_A.rows(); }

  /**
   * @brief Returns the column count of the sparse matrix.
   * @return Number of columns in the coefficient matrix as unsigned int.
   */
  unsigned column_count() const { return this->_A.cols(); }

  /**
   * @brief Solves the sparse linear system using Eigen's Sparse QR
   * decomposition.
   * @return Solution vector of the linear system of type Eigen::VectorXd.
   */
  Vector eigen_solve() const;

  /**
   * @brief Generates a random banded sparse linear system with full-rank.
   * @param rng Random number generator.
   * @param dim Dimension of the square matrix to generate.
   * @param bandwidth Bandwidth of the matrix (too large bandwidths are okay and
   * are handled by function).
   * @return A random SparseLinearSystem object with a full-rank matrix of
   * specified dimensions and bandwidth.
   */
  static SparseLinearSystem generate_random_banded_regular(std::mt19937 &rng,
                                                           unsigned dim,
                                                           unsigned bandwidth);

  /**
   * @brief Reads a sparse linear system from an input stream.
   * @param in_stream Input stream containing matrix and vector data.
   * @return A SparseLinearSystem initialized with values from the stream.
   * @details The input stream should provide data in the following format:
   *          - The first line contains three unsigned integers:
   *            1. `nnz`: number of non-zero elements in the matrix,
   *            2. `rows`: number of rows in the matrix,
   *            3. `cols`: number of columns in the matrix.
   *          - The next `nnz` lines each contain three values representing a
   * matrix entry:
   *            1. `row` (unsigned): row index of the entry,
   *            2. `col` (unsigned): column index of the entry,
   *            3. `value` (double): value at that matrix position.
   *          - The final `rows` lines contain values representing the
   * right-hand side vector (RHS), one double per line.
   * @note Expects the matrix data to be in triplet format, followed by the RHS
   * vector values.
   */
  static SparseLinearSystem read_from_stream(std::istream &in_stream);

  /**
   * @brief Writes the sparse linear system to an output stream.
   * @param out_stream Output stream to which matrix and vector data will be
   * written.
   * @details The output stream will contain data in the following format:
   *          - The first line contains the number of non-zero elements (`nnz`)
   * in the matrix.
   *          - The second line contains two unsigned integers:
   *            1. `rows`: number of rows in the matrix,
   *            2. `cols`: number of columns in the matrix.
   *          - The next `nnz` lines each represent a non-zero matrix entry
   * with:
   *            1. `row` (unsigned): row index of the entry,
   *            2. `col` (unsigned): column index of the entry,
   *            3. `value` (double): value at that matrix position.
   *          - The final `rows` lines contain values of the right-hand side
   * vector (RHS), one double per line.
   * @note Outputs the matrix in triplet format, followed by the values of the
   * RHS vector.
   */
  void write_to_stream(std::ostream &out_stream) const;
};

/**
 * A sparse representation of a LSE where the coefficient matrix is a banded
 * matrix.
 */
class BandedLinearSystem {
 private:
  unsigned _dim;
  unsigned _bandwidth;
  /**
   * Refer to the .A_data() method for an explanation of the representation.
   */
  std::vector<double> _A_data;
  /**
   * The RHS of this LSE.
   */
  Vector _b;

  BandedLinearSystem(const unsigned dim, const unsigned bandwidth,
                     const std::vector<double> &A_data, const Vector b)
      : _dim(dim), _bandwidth(bandwidth), _A_data(A_data), _b(b) {}

 public:
  /**
   * Returns the set bandwidth of the banded matrix that is coefficient matrix
   * of this LSE.
   */
  unsigned bandwidth() const { return this->_bandwidth; }

  /**
   * Returns the number of columns (which is simultaneously the number of rows)
   * of the coefficient matrix of this LSE.
   */
  unsigned dim() const { return this->_dim; }
  /**
   * The (potentially) non-zero entries of the coefficient matrix of this LSE.
   * Stored in a row-major format.
   *
   * Take each row of the coefficient matrix and consider that subarray of its
   * elements that is within the set bandwidth away from the main diagonal. The
   * vector returned here consists of the concatenation of all these subarrays
   * from row 0 to row dim - 1.
   */
  const std::vector<double> &A_data() const { return this->_A_data; }
  /**
   * The RHS of this LSE.
   */
  const Vector &b() const { return this->_b; }

  /**
   * Generate a random LSE with a banded coefficient matrix of a given dimension
   * (dim) and bandwidth. The coefficient values (and those of the RHS) are
   * chosen i.i.d. from the uniform distribution on [-1, 1].
   */
  static BandedLinearSystem generate_random_regular(std::mt19937 &rng,
                                                    unsigned dim,
                                                    unsigned bandwidth);

  SparseLinearSystem to_sparse_system() const;
};

#endif  // SPARSE_HPP
