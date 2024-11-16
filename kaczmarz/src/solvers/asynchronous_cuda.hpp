#ifndef ASYNC_CUDA_HPP
#define ASYNC_CUDA_HPP

#include "common.hpp"

KaczmarzSolverStatus invoke_asynchronous_solver_gpu(const int* h_A_outer,
const int* h_A_inner,
const double* h_A_values,
const double* h_b,
double* h_x,
double* h_sq_norms,
const unsigned rows, const unsigned cols, const unsigned nnz,
                              
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads);

                                              #endif  // ASYNC_CUDA_HPP