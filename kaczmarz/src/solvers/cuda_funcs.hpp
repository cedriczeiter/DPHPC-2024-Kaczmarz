#ifndef CUDA_FUNCS_HPP
#define CUDA_FUNCS_HPP

/**
 * @brief Computes the dot product of a matrix and a vector.
 *
 * This function computes the dot product of a matrix A and a vector x. The
 * result is stored in the h_result vector.
 *
 * @param h_A The matrix A.
 * @param h_x The vector x.
 * @param h_result The vector to store the result.
 * @param rows The number of rows in the matrix A.
 * @param cols The number of columns in the matrix A.
 */
void dot_product_cuda(const double *h_A, const double *h_x, double *h_result, const unsigned rows, const unsigned cols);

/**
 * @brief Computes the squared norm of a matrix.
 *
 * This function computes the squared norm of a matrix A. The result is stored
 * in the h_result vector.
 *
 * @param h_A The matrix A.
 * @param h_result The vector to store the result.
 * @param rows The number of rows in the matrix A.
 * @param cols The number of columns in the matrix A.
 */
void squared_norm_cuda(const double *h_A, double *h_result, const unsigned rows, const unsigned cols);

#endif  // CUDA_FUNCS_HPP