#ifndef KACZMARZ_BANDED_HPP
#define KACZMARZ_BANDED_HPP

#include "../kaczmarz_serial/kaczmarz_common.hpp"
#include "linear_systems/sparse.hpp"

KaczmarzSolverStatus kaczmarz_banded_2_cpu_threads(
    const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations,
    double precision);

KaczmarzSolverStatus kaczmarz_banded_serial(const BandedLinearSystem& lse,
                                            Eigen::VectorXd& x,
                                            unsigned max_iterations,
                                            double precision);

#endif  // KACZMARZ_BANDED_HPP
