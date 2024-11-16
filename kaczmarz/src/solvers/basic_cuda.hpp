#ifndef BASIC_CUDA_HPP
#define BASIC_CUDA_HPP

#include "common.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
/**
 * @brief Not implemented yet.
 */
KaczmarzSolverStatus dense_kaczmarz_cuda(const DenseLinearSystem& lse, double* x,
                                         unsigned max_iterations, double precision,
                                         std::vector<double>& times_residuals,
                                         std::vector<double>& residuals,
                                         std::vector<int>& iterations,
                                         const int convergence_step_rate);

double invoke_dense_kaczmarz_update(const DenseLinearSystem &lse, double *x,
                             const unsigned rows, const unsigned cols);

#endif // BASIC_CUDA_HPP