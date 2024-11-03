#ifndef BASIC_HPP
#define BASIC_HPP

#include "common.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
/**
 * @brief Solves the Ax = b LSE by running the serial Kaczmarz algorithm with
 * in-order row updates.
 *
 * This function attempts to solve the system of linear equations Ax = b
 * using an iterative serial Kaczmarz method. The algorithm iteratively refines
 * the solution vector x.
 *
 * @param lse The linear system to solve.
 * @param x Pointer to the initial guess for the solution vector x. The solution
 * of Ax=b will be stored here.
 * @param max_iterations Maximum number of iterations to perform. Before the
 * algorithm forced to end.
 * @param precision Convergence precision threshold. The algorithm stops if the
 * correction is less than this value.
 * @return KaczmarzSolverStatus indicating the status of the solver
 */
KaczmarzSolverStatus dense_kaczmarz(const DenseLinearSystem& lse, double* x,
                                    unsigned max_iterations, double precision,
                                    std::vector<double>& times_residuals,
                                    std::vector<double>& residuals,
                                    std::vector<int>& iterations,
                                    const int convergence_step_rate);

/**
 * @brief Analogous to dense_kaczmarz but operates on sparse LSEs.
 */
KaczmarzSolverStatus sparse_kaczmarz(const SparseLinearSystem& lse,
                                     Eigen::VectorXd& x,
                                     unsigned max_iterations, double precision,
                                     std::vector<double>& times_residuals,
                                     std::vector<double>& residuals,
                                     std::vector<int>& iterations,
                                     const int convergence_step_rate);

#endif  // BASIC_HPP
