#ifndef BANDED_HPP
#define BANDED_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"

void kaczmarz_banded_openmp_grouping1(const BandedLinearSystem& lse, Vector& x,
                                      unsigned iterations,
                                      unsigned thread_count);

void kaczmarz_banded_openmp_grouping2(const BandedLinearSystem& lse, Vector& x,
                                      unsigned iterations,
                                      unsigned thread_count);

void kaczmarz_banded_serial_naive(const BandedLinearSystem& lse, Vector& x,
                                  unsigned iterations);

void kaczmarz_banded_serial_interleaved(const BandedLinearSystem& lse,
                                        Vector& x, unsigned iterations);

KaczmarzSolverStatus kaczmarz_banded_cuda(const BandedLinearSystem& lse,
                                          Eigen::VectorXd& x,
                                          unsigned max_iterations,
                                          double precision);

#endif  // BANDED_HPP
