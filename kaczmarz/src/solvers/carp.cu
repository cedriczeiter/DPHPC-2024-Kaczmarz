#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <set>

#include "carp_utils.hpp"
#include "common.hpp"

// IMPORTANT: ONLY WORKS ON SQUARE MATRICES ATM AND IF ROWS_PER_THREAD DIVIDES
// TOTAL ROWS(dim)

KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned dim,
    const unsigned nnz, const unsigned max_iterations, const double precision,
    const unsigned max_nnz_in_row, const double b_norm) {
  // define some variables
  bool converged = false;
  const unsigned total_threads = dim / ROWS_PER_THREAD;

  // allocate move squared norms on device
  double *d_sq_norms;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_sq_norms, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpy(d_sq_norms, h_sq_norms, dim * sizeof(double),
                            cudaMemcpyHostToDevice));

  // move x to device
  double *d_x;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, dim * sizeof(double)));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_x, h_x, dim * sizeof(double), cudaMemcpyHostToDevice));

  // move A to device
  int *d_A_outer;
  int *d_A_inner;
  double *d_A_values;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_outer, (dim + 1) * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_inner, nnz * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_values, nnz * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_outer, h_A_outer, (dim + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_inner, h_A_inner, nnz * sizeof(int),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                            cudaMemcpyHostToDevice));

  // we need to know which values in x are affected by which thread; thats what
  // the below code is for
  int *h_affected = new int[dim]();
  for (unsigned row = 0; row < dim; row++) {
    int h_A_outer_row = h_A_outer[row];
    int h_A_outer_row_plus_one = h_A_outer[row + 1];
    for (unsigned i = h_A_outer_row; i < h_A_outer_row_plus_one; i++) {
      h_affected[h_A_inner[i]]++;
    }
  }

  // move affects to device
  int *d_affected;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_affected, dim * sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_affected, h_affected, dim * sizeof(int),
                            cudaMemcpyHostToDevice));

  // move b to device
  double *d_b;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, dim * sizeof(double)));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_b, h_b, dim * sizeof(double), cudaMemcpyHostToDevice));

  // move p, r, q and intermediate storage to device
  double *d_p;
  double *d_r;
  double *d_q;
  double *d_intermediate;
  double *d_intermediate_two;
  double *d_zero;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_p, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_r, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_q, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_intermediate, dim * sizeof(double)));
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_intermediate_two, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_zero, dim * sizeof(double)));
  CUDA_SAFE_CALL(cudaMemset((void **)d_zero, 0, dim * sizeof(double)));

  // calculate nr of blocks and threads
  const int blocks =
      (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // solve LSE:
  // init stuff
  const double relaxation = 1.0;
  double residual = 1.;  // init value, will be overwritten as soon as we check
                         // for convergence
  dcswp(d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_x, relaxation,
        d_affected, total_threads, d_r, d_intermediate, blocks, max_nnz_in_row);
  copy_gpu(d_r, d_p, dim);

  for (int iter = 0; iter < max_iterations; iter++) {
    // calculate residual every L_RESIDUAL iterations
    if (iter % L_RESIDUAL == 0) {
      residual =
          get_residual(h_x, h_b, d_x, h_A_outer, h_A_inner, h_A_values, dim);
      // debugging output
      std::cout << "Iteration: " << iter << " out of " << max_iterations
                << " , Residual/B_norm: " << residual / b_norm << std::endl;
      // check for convergence
      if (residual / b_norm < precision) {
        converged = true;
        break;  // stop all the iterations
      }
    }

    // the actual calculation begin here
    dcswp(d_A_outer, d_A_inner, d_A_values, d_zero, dim, d_sq_norms, d_p,
          relaxation, d_affected, total_threads, d_intermediate,
          d_intermediate_two, blocks, max_nnz_in_row);
    add_gpu(d_p, d_intermediate, d_q, -1., dim);
    const double sq_norm_r_old = dot_product_gpu(d_r, d_r, d_intermediate, dim);
    const double dot_r_p = dot_product_gpu(d_p, d_q, d_intermediate, dim);
    /*if (dot_r_p < 1e-30) {  // if dot_r_p too small, algorithm is in flat
    region
                            // and cannot move further. Either we converged, or
                            // we need to continue with a different algorithm
      residual =
          get_residual(h_x, h_b, d_x, h_A_outer, h_A_inner, h_A_values, dim);
      break;
    }*/  //this does not work. It prevents the algorithm fom converging for PDE 2
    const double alpha = sq_norm_r_old / dot_r_p;
    if (std::isinf(alpha) ||
        std::isnan(alpha)) {  // another safeguard to see if converged, nothing
                              // more to the algorithm can do
      residual =
          get_residual(h_x, h_b, d_x, h_A_outer, h_A_inner, h_A_values, dim);
      break;
    }
    add_gpu(d_x, d_p, d_x, alpha, dim);
    add_gpu(d_r, d_q, d_r, -alpha, dim);
    const double beta =
        dot_product_gpu(d_r, d_r, d_intermediate, dim) / sq_norm_r_old;
    add_gpu(d_r, d_p, d_p, beta, dim);
  }

  // free memory
  cudaFree(d_x);
  cudaFree(d_affected);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);
  cudaFree(d_p);
  cudaFree(d_r);
  cudaFree(d_q);
  cudaFree(d_intermediate);
  cudaFree(d_intermediate_two);
  cudaFree(d_zero);
  // check for convergence
  if (converged) {
    return KaczmarzSolverStatus::Converged;
  } else {
    return KaczmarzSolverStatus::OutOfIterations;
  }
}