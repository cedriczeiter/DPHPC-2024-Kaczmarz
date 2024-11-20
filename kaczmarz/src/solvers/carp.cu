#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <set>

#include "common.hpp"
#include "carp_utils.hpp"

#define L_RESIDUAL 100
#define ROWS_PER_THREAD 10
#define LOCAL_RUNS_PER_THREAD 1
#define THREADS_PER_BLOCK 32

// IMPORTANT: ONLY WORKS ON SQUARE MATRICES ATM AND IF ROWS_PER_THREAD DIVIDES
// TOTAL ROWS

KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned rows,
    const unsigned cols, const unsigned nnz, const unsigned max_iterations,
    const double precision, const unsigned max_nnz_in_row, const double b_norm) {
  // check if matrix is square
  assert(rows == cols);
  const unsigned dim = rows;

  // define some variables
  bool converged = false;
  const unsigned total_threads = rows / ROWS_PER_THREAD;

  // allocate move squared norms on device
  double *d_sq_norms;
  cudaMalloc((void **)&d_sq_norms, rows * sizeof(double));
  cudaMemcpy(d_sq_norms, h_sq_norms, rows * sizeof(double),
             cudaMemcpyHostToDevice);

  // move x to device
  double *d_x;
  cudaMalloc((void **)&d_x, cols * sizeof(double));
  cudaMemcpy(d_x, h_x, cols * sizeof(double), cudaMemcpyHostToDevice);

  // move A to device
  int *d_A_outer;
  int *d_A_inner;
  double *d_A_values;
  cudaMalloc((void **)&d_A_outer, (rows + 1) * sizeof(int));
  cudaMalloc((void **)&d_A_inner, nnz * sizeof(int));
  cudaMalloc((void **)&d_A_values, nnz * sizeof(double));
  cudaMemcpy(d_A_outer, h_A_outer, (rows + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_inner, h_A_inner, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
             cudaMemcpyHostToDevice);

  // we need to know which values in x are affected by which thread; thats what
  // the below code is for
  std::vector<std::set<unsigned>> affects(
      rows);  // coding: affects[j]: the x at position j is affected by thread
              // in set
  for (unsigned k = 0; k < rows; k++) {
    for (unsigned i = h_A_outer[k]; i < h_A_outer[k + 1]; i++) {
      const unsigned thread = (unsigned)(h_A_inner[i] / ROWS_PER_THREAD);
      affects.at(k).insert(thread);
    }
  }
  const int affects_size = affects.size();

  // move affects to device memory, set to -1 if no thread affects the value
  // (default value)
  int *h_affected = new int[(total_threads + 1) * rows];
  std::memset(h_affected, -1, rows * (total_threads + 1) * sizeof(int));
  // Translate it to a 1D array
  for (int k = 0; k < affects_size; k++) {
    unsigned counter = 0;
    for (const auto &thread : affects[k]) {
      h_affected[k * (total_threads + 1) + counter] = thread;
      counter++;
    }
  }

  // move affects to device
  int *d_affected;
  cudaMalloc((void **)&d_affected, rows * (total_threads + 1) * sizeof(int));
  cudaMemcpy(d_affected, h_affected, rows * (total_threads + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  // move b to device
  double *d_b;
  cudaMalloc((void **)&d_b, cols * sizeof(double));
  cudaMemcpy(d_b, h_b, cols * sizeof(double), cudaMemcpyHostToDevice);

  //move p, r, q and intermediate storage to device
  double *d_p;
  double *d_r;
  double *d_q;
  double *d_intermediate;
  double *d_zero;
  cudaMalloc((void**)&d_p, dim*sizeof(double));
  cudaMalloc((void**)&d_r, dim*sizeof(double));
  cudaMalloc((void**)&d_q, dim*sizeof(double));
  cudaMalloc((void**)&d_intermediate, dim*sizeof(double));
  cudaMalloc((void**)&d_zero, dim*sizeof(double));
  cudaMemset((void **)d_zero, 0, dim*sizeof(double));

  // move X to device
  double *d_X;
  cudaMalloc((void **)&d_X, total_threads * cols * sizeof(double));
  cudaMemset((void **)d_X, 0, total_threads * cols * sizeof(double));

  // calculate nr of blocks and threads
  const int blocks =
      (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // solve LSE
  double base_residual = 0;

  //init stuff
  const double relaxation = 1.0;
  dcswp(d_A_outer, d_A_inner,
                     d_A_values, d_b,
                    dim,
                    d_sq_norms, d_x, d_X,
                     relaxation, d_affected, total_threads, d_r, blocks);
  copy_gpu(d_r, d_p, dim);

  for (int iter = 0; iter < max_iterations; iter++) {
    // calculate residual every L_RESIDUAL iterations
    if (iter % L_RESIDUAL == 0) {
      cudaMemcpy(h_x, d_x, cols * sizeof(double), cudaMemcpyDeviceToHost);
      double residual = 0.0;
      // Calulate residual
      for (unsigned i = 0; i < rows; i++) {
        double dot_product = 0.0;
        for (unsigned j = h_A_outer[i]; j < h_A_outer[i + 1]; j++) {
          dot_product += h_A_values[j] * h_x[h_A_inner[j]];
        }
        residual += (dot_product - h_b[i]) * (dot_product - h_b[i]);
      }
      residual = sqrt(residual);

      // First residual is the base residual
      if (iter == 0) {
        base_residual = residual;
      }

      // debugging output
      printf("Iteration: %d out of %d, Residual/B_norm: %f\n", iter,
             max_iterations, residual/b_norm);

      // check for convergence
      if (residual / base_residual < precision) {
        converged = true;
        break;  // stop all the iterations
      }
    }

    // the real work begins here
    dcswp(d_A_outer, d_A_inner,
                     d_A_values, d_zero,
                    dim,
                    d_sq_norms, d_p, d_X,
                     relaxation, d_affected, total_threads, d_intermediate, blocks);
    add_gpu(d_p, d_intermediate, d_q, -1., dim);
    const double sq_norm_r_old = dot_product_gpu(d_r, d_r, d_intermediate, dim);
    const double alpha = sq_norm_r_old/dot_product_gpu(d_p, d_q, d_intermediate, dim);
    add_gpu(d_x, d_p, d_x, alpha, dim);
    add_gpu(d_r, d_q, d_r, -alpha, dim);
    const double beta = dot_product_gpu(d_r, d_r, d_intermediate, dim)/sq_norm_r_old;
    add_gpu(d_r, d_p, d_p, beta, dim);
  }

  // free memory
  cudaFree(d_x);
  cudaFree(d_X);
  cudaFree(d_affected);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);

  // check for convergence
  if (converged) {
    return KaczmarzSolverStatus::Converged;
  } else {
    return KaczmarzSolverStatus::OutOfIterations;
  }
}