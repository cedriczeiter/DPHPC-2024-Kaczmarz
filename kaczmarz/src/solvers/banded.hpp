#ifndef BANDED_HPP
#define BANDED_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"

/**
 * Run an implementation of the Kaczmarz solver that is parallelized for two CPU
 * cores using OpenMP.
 *
 * This implementation can only solve LSEs with banded coefficient matrices
 * because it makes use of the information of which rows are orthogonal.
 */
KaczmarzSolverStatus kaczmarz_banded_2_cpu_threads(
    const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations,
    double precision);

/**
 * Run an implementation of the Kaczmarz solver which is specialized for LSEs
 * with banded coefficient matrices. This implementation does not attempt any
 * parallelization.
 *
 * The purpose of this implementation is to be a reference point when evaluating
 * the performance of `kaczmarz_banded_2_cpu_threads`. This and that
 * implementation are almost the same. This implementation just doesn't take
 * advantage of a parallelization opportunity, which
 * `kaczmarz_banded_2_cpu_threads` exploits.
 */
KaczmarzSolverStatus kaczmarz_banded_serial(const BandedLinearSystem& lse,
                                            Eigen::VectorXd& x,
                                            unsigned max_iterations,
                                            double precision);

KaczmarzSolverStatus kaczmarz_banded_cuda(const BandedLinearSystem& lse,
                                          Eigen::VectorXd& x,
                                          unsigned max_iterations,
                                          double precision);

#endif  // BANDED_HPP
