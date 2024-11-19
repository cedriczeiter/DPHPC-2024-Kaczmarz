#ifndef CARP_CUDA_HPP
#define CARP_CUDA_HPP

#include "common.hpp"

KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned rows,
    const unsigned cols, const unsigned nnz, const unsigned max_iterations,
    const double precision, const unsigned max_nnz_in_row);

#endif  // CARP_CUDA_HPP