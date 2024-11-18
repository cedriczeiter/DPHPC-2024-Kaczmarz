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

bool computeRowSums(const std::vector<double>& A, const std::vector<double>& x, 
                                        std::vector<double>& dot_product, std::vector<double>& row_sq_norm, 
                                        int rows, int cols);
#endif // BASIC_CUDA_HPP