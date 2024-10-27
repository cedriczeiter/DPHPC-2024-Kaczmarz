#ifndef TYPES_HPP
#define TYPES_HPP

#include <Eigen/SparseCore>

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    DenseMatrix;
typedef Eigen::VectorXd Vector;

#endif  // TYPES_HPP
