#ifndef BANDED_HPP
#define BANDED_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"

KaczmarzSolverStatus kaczmarz_banded_2_cpu_threads(
    const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations,
    double precision);

KaczmarzSolverStatus kaczmarz_banded_serial(const BandedLinearSystem& lse,
                                            Eigen::VectorXd& x,
                                            unsigned max_iterations,
                                            double precision);

#endif  // BANDED_HPP
