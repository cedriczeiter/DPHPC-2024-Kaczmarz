#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include <cassert>


#include "common.hpp"

__global__ void solve_async(const int *A_outerIndex, const int *A_innerIndex,
                            const double *A_values, const double *b,
                            const unsigned rows, const unsigned cols,
                            const double *sq_norms, double *x,
                            const unsigned max_iterations,
                            const unsigned runs_before_sync, bool *converged,
                            const unsigned L, const double precision,
                            const unsigned num_threads) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(13, tid, 0, &state);

  // we want each thread to operate on set of own rows (mutually almost
  // exclusive)
  const unsigned n_own_rows = (rows / num_threads) + 1;
  const unsigned thread_offset = n_own_rows * tid;

  for (unsigned iter = 0; iter < max_iterations; iter++) {
    for (unsigned inner_iter = 0; inner_iter < runs_before_sync; inner_iter++) {
      // get random row
      const unsigned k = thread_offset + (curand(&state) % n_own_rows);

      if (k >= rows) continue;

      // compute dot product row * x
      double dot_product = 0.;
      for (unsigned i = A_outerIndex[k]; i < A_outerIndex[k + 1]; i++) {
        const double x_value = atomicAdd(&x[A_innerIndex[i]], 0.);
        dot_product += A_values[i] * x_value;
      }

      const double update_coeff = ((b[k] - dot_product) / sq_norms[k]);

      // update x
      for (unsigned i = A_outerIndex[k]; i < A_outerIndex[k + 1]; i++) {
        const double update = update_coeff * A_values[i];
        atomicAdd(&x[A_innerIndex[i]], update);
      }
    }
    // sync all threads to guarantee convergence
    __syncthreads();
    if (iter % L == 0 && iter > 0 && tid == 0) {
      double residual = 0.0;
      for (unsigned i = 0; i < rows; i++) {
        double dot_product = 0.0;
        for (unsigned j = A_outerIndex[i]; j < A_outerIndex[i + 1]; j++) {
          dot_product += A_values[j] * atomicAdd(&x[A_innerIndex[j]], 0);
        }
        residual += (dot_product - b[i]) * (dot_product - b[i]);
      }
      residual = sqrt(residual);

      printf("Residual: %f\n", residual);

      if (residual < precision) {
        *converged = true;
      }
    }
    __threadfence();
    __syncthreads();
    if (*converged) break;
  }
}

KaczmarzSolverStatus invoke_asynchronous_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned rows,
    const unsigned cols, const unsigned nnz,

    const unsigned max_iterations, const double precision,
    const unsigned num_threads) {
  assert(num_threads <=
         rows);  // necessary for allowing each thread to have local rows

  const unsigned L = 5000;  // we check for convergence every L steps
  const unsigned runs_before_sync = 500;

  // allocate move squared norms on device
  double *d_sq_norms;
  cudaMalloc((void **)&d_sq_norms, rows * sizeof(double));
  cudaMemcpy(d_sq_norms, h_sq_norms, rows * sizeof(double),
             cudaMemcpyHostToDevice);

  // move x to device
  double *d_x;
  cudaMalloc((void **)&d_x, cols * sizeof(double));
  cudaMemcpy(d_x, h_x, cols * sizeof(double), cudaMemcpyHostToDevice);

  // create convergence flag and move to device
  bool *h_converged = new bool(false);
  bool *d_converged;
  cudaMalloc((void **)&d_converged, sizeof(bool));
  cudaMemcpy(d_converged, h_converged, sizeof(bool), cudaMemcpyHostToDevice);

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

  // move b to device
  double *d_b;
  cudaMalloc((void **)&d_b, cols * sizeof(double));
  cudaMemcpy(d_b, h_b, cols * sizeof(double), cudaMemcpyHostToDevice);

  // solve LSE
  std::cout << "Calling kernel\n";
  solve_async<<<1, num_threads>>>(
      d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x,
      max_iterations, runs_before_sync, d_converged, L, precision, num_threads);
  cudaDeviceSynchronize();
  std::cout << "Kernel done\n";

  // copy back x and convergence
  cudaMemcpy(h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_x, d_x, cols * sizeof(double), cudaMemcpyDeviceToHost);

  // free memory
  cudaFree(d_converged);
  cudaFree(d_x);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);

  // check for convergence
  if (*h_converged) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::OutOfIterations;
}