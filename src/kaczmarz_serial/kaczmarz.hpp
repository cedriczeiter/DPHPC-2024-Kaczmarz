#ifndef KACZMARZ_HPP
#define KACZMARZ_HPP

#include "kaczmarz_common.hpp"

/**
 * @brief Solves the Ax = b LSE by running the serial Kaczmarz algorithm with in-order row updates.
 *
 * This function attempts to solve the system of linear equations Ax = b
 * using an iterative serial Kaczmarz method. The algorithm iteratively refines
 * the solution vector x.
 *
 * @param A Pointer to the matrix A (stored in row-major order). Part of the linear system (Ax=b) to solve.
 * @param b Pointer to the vector b. Part of the linear system (Ax=b) to solve.
 * @param x Pointer to the initial guess for the solution vector x. The solution of Ax=b will be stored here.
 * @param rows Number of rows in the matrix A. (And elements in b)
 * @param cols Number of columns in the matrix A.
 * @param max_iterations Maximum number of iterations to perform. Before the algorithm forced to end.
 * @param precision Convergence precision threshold. The algorithm stops if the correction is less than this value.
 * @return KaczmarzSolverStatus indicating the status of the solver:
 *         - KaczmarzSolverStatus::Converged: The algorithm has converged to a solution.
 *         - KaczmarzSolverStatus::ZeroNormRow: A row in the matrix A has zero norm, making the system unsolvable.
 *         - KaczmarzSolverStatus::OutOfIterations: The algorithm reached the maximum number of iterations without converging.
 */
KaczmarzSolverStatus kaczmarz_solver(const double *A, const double *b, double *x, unsigned rows, unsigned cols, unsigned max_iterations, double precision);

#endif // KACZMARZ_HPP
