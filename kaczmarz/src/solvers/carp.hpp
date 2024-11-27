#ifndef CARP_HPP
#define CARP_HPP
#include <unistd.h>

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

/**
 * @brief Solves a sparse linear system using the Kaczmarz method on a GPU.
 *
 * @param lse The sparse linear system to be solved.
 * @param x The vector to store the solution.
 * @param max_iterations The maximum number of iterations to perform.
 * @param precision The precision required for the solution.
 * @return The status of the solver after completion.
 */
KaczmarzSolverStatus carp_gpu(const SparseLinearSystem& lse, Vector& x,
                              const unsigned max_iterations,
                              const double precision);

#endif
