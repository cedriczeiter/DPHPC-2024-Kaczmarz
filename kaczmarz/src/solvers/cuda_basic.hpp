#ifndef CUDA_BASIC_HPP
#define CUDA_BASIC_HPP

#include "common.hpp"
#include "dense.hpp"
#include "sparse.hpp"
/**
 * @brief Solves the Ax = b LSE by running the serial Kaczmarz algorithm with
 * in-order row updates. Works on CUDA.
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
 * @param times_residuals Vector to store the times at which residuals are
 * recorded.
 * @param residuals Vector to store the residuals at each recorded time.
 * @param iterations Vector to store the iteration counts at each recorded time.
 * @param convergence_step_rate The rate at which convergence is checked.
 * @return KaczmarzSolverStatus indicating the status of the solver
 */
KaczmarzSolverStatus dense_kaczmarz_cuda(const DenseLinearSystem& lse, double* x,
                                    unsigned max_iterations, double precision,
                                    std::vector<double>& times_residuals,
                                    std::vector<double>& residuals,
                                    std::vector<int>& iterations,
                                    const int convergence_step_rate);

/**
 * @brief Analogous to dense_kaczmarz but operates on sparse LSEs. Works on CUDA.
 *
 * @param lse The sparse linear system to solve.
 * @param x The initial guess for the solution vector x. The solution of Ax=b
 * will be stored here.
 * @param max_iterations Maximum number of iterations to perform.
 * @param precision Convergence precision threshold.
 * @param times_residuals Vector to store the times at which residuals are
 * recorded.
 * @param residuals Vector to store the residuals at each recorded time.
 * @param iterations Vector to store the iteration counts at each recorded time.
 * @param convergence_step_rate The rate at which convergence is checked.
 * @return KaczmarzSolverStatus indicating the status of the solver
 */
KaczmarzSolverStatus sparse_kaczmarz_cuda(const SparseLinearSystem& lse,
                                     Eigen::VectorXd& x,
                                     unsigned max_iterations, double precision,
                                     std::vector<double>& times_residuals,
                                     std::vector<double>& residuals,
                                     std::vector<int>& iterations,
                                     const int convergence_step_rate);

#endif  // CUDA_BASIC_HPP
