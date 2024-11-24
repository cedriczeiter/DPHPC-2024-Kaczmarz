#ifndef CARP_CUDA_HPP
#define CARP_CUDA_HPP

#include "common.hpp"

/**
 * @brief Invokes the CARP solver on a GPU.
 *
 * This function solves a system of linear equations using the Kaczmarz method
 * on a GPU.
 *
 * @param h_A_outer Pointer to the outer indices of the sparse matrix in CSR
 * format.
 * @param h_A_inner Pointer to the inner indices of the sparse matrix in CSR
 * format.
 * @param h_A_values Pointer to the non-zero values of the sparse matrix in CSR
 * format.
 * @param h_b Pointer to the right-hand side vector.
 * @param h_x Pointer to the solution vector.
 * @param h_sq_norms Pointer to the squared norms of the rows of the matrix.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @param nnz Number of non-zero elements in the matrix.
 * @param max_iterations Maximum number of iterations to perform.
 * @param precision Desired precision for the solution.
 * @param max_nnz_in_row Maximum number of non-zero elements in any row of the
 * matrix.
 * @param b_norm Norm of the right-hand side vector.
 * @return Status of the solver after execution.
 */

KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned dim, const unsigned nnz, const unsigned max_iterations,
    const double precision, const unsigned max_nnz_in_row, const double b_norm);

#endif  // CARP_CUDA_HPP