#ifndef ASYNC_HPP
#define ASYNC_HPP

#include <unistd.h>

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

/**
 * @brief Solves a sparse linear system using the asynchronous Kaczmarz method
 * on a GPU.
 *
 * @param lse The sparse linear system to be solved.
 * @param x The initial guess for the solution vector, which will be updated
 * with the solution.
 * @param max_iterations The maximum number of iterations to perform.
 * @param precision The desired precision for the solution.
 * @param num_threads The number of threads to use for the computation.
 * @return The status of the solver after completion.
 */
KaczmarzSolverStatus asynchronous_gpu(const SparseLinearSystem &lse, Vector &x,
                                      const unsigned max_iterations,
                                      const double precision,
                                      const unsigned num_threads);

/**
 * @brief Solves a sparse linear system using the asynchronous Kaczmarz method
 * on a CPU.
 *
 * @param lse The sparse linear system to be solved.
 * @param x The initial guess for the solution vector, which will be updated
 * with the solution.
 * @param max_iterations The maximum number of iterations to perform.
 * @param precision The desired precision for the solution.
 * @param num_threads The number of threads to use for the computation.
 * @return The status of the solver after completion.
 */
KaczmarzSolverStatus asynchronous_cpu(const SparseLinearSystem &lse, Vector &x,
                                      const unsigned max_iterations,
                                      const double precision,
                                      const unsigned num_threads);

#endif  // ASYNC_HPP