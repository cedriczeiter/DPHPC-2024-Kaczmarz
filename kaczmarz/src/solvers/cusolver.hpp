#ifndef CUDA_NATIVE_SOLVER_HPP
#define CUDA_NATIVE_SOLVER_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

/**
 * @brief Solves a sparse linear system using the CuDSS library.
 * library.
 *
 * @param lse The sparse linear system to be solved.
 * @param x The vector to store the solution.
 * @param max_iterations The maximum number of iterations to perform.
 * @param precision The precision required for the solution.
 * @return The status of the solver after completion.
 */
KaczmarzSolverStatus cusolver(const SparseLinearSystem& lse, Vector& x,
                              const unsigned max_iterations,
                              const double precision);

#endif  // CUDA_NATIVE_SOLVER_HPP