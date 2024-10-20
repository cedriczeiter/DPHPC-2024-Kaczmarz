#ifndef KACZMARZ_RANDOM_HPP
#define KACZMARZ_RANDOM_HPP

#include "kaczmarz_common.hpp"

#include <vector>

/**
 * @brief Solves the Ax = b linear system using the randomized Kaczmarz
 * algorithm.
 *
 * This function attempts to solve the system of linear equations Ax = b
 * using an iterative randomized Kaczmarz method. The algorithm iteratively
 * refines the solution vector x by randomly selecting rows based on their
 * squared norms.
 *
 * @param A Pointer to the matrix A (stored in row-major order). Part of the
 * linear system (Ax=b) to solve.
 * @param b Pointer to the vector b. Part of the linear system (Ax=b) to solve.
 * @param x Pointer to the initial guess for the solution vector x. The solution
 * of Ax=b will be stored here.
 * @param rows Number of rows in the matrix A. (And elements in b)
 * @param cols Number of columns in the matrix A.
 * @param max_iterations Maximum number of iterations to perform. Before the
 * algorithm is forced to end.
 * @param precision Convergence precision threshold. The algorithm stops if the
 * correction is less than this value.
 * @return KaczmarzSolverStatus indicating the status of the solver:
 *         - KaczmarzSolverStatus::Converged: The algorithm has converged to a
 * solution.
 *         - KaczmarzSolverStatus::ZeroNormRow: A row in the matrix A has zero
 * norm, making the system unsolvable.
 *         - KaczmarzSolverStatus::OutOfIterations: The algorithm reached the
 * maximum number of iterations without converging.
 */
KaczmarzSolverStatus kaczmarz_random_solver(const double *A, const double *b,
                                            double *x, unsigned rows,
                                            unsigned cols,
                                            unsigned max_iterations,
                                            double precision);

/**
 * @brief Helper function to select a random row based on its squared norm.
 *
 * This function selects a row from the matrix A based on the row's squared
 * norm. The selection probability of each row is proportional to its squared
 * norm.
 *
 * @param row_norms Pointer to the array containing squared norms of each row in
 * A.
 * @param num_rows The number of rows in the matrix A.
 * @return The index of the selected row.
 */
unsigned random_row_selection(const double *row_norms, unsigned num_rows);

#endif // KACZMARZ_RANDOM_HPP
