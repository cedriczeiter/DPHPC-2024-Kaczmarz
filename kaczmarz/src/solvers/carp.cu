#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include <cassert>


#include "common.hpp"

//IMPORTANT: ONLY WORKS ON SQUARE MATRICES ATM

__global__ void step(const int *A_outerIndex, const int *A_innerIndex,
                            const double *A_values, const double *b,
                            const unsigned rows, const unsigned cols,
                            const double *sq_norms, double *x, double *X) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < rows){
    //perform one update step for assigned row

    // compute dot product row * x
    double dot_product = 0.;
    for (unsigned i = A_outerIndex[tid]; i < A_outerIndex[tid + 1]; i++) {
        const double x_value = x[A_innerIndex[i]];
        dot_product += A_values[i] * x_value;
    }

    //calculate update
    const double update_coeff = ((b[tid] - dot_product) / sq_norms[tid]);

    // save update for x in global matrix, will be used in average step
    for (unsigned i = A_outerIndex[tid]; i < A_outerIndex[tid + 1]; i++) {
        const double update = update_coeff * A_values[i];
        X[tid*cols + A_innerIndex[i]] = update;
        //printf("Update: %f\n", update);
    }
  }
}

__global__ void update(const int *A_outerIndex, const int *A_innerIndex,
                            const double *A_values, const double *b,
                            const unsigned rows, const unsigned cols,
                            const double *sq_norms, double *x, double *X) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < cols){
    //sum up updates for assigned entry
    double sum = 0;
    double count = 0;
    for (int i = 0; i < cols; i++){
        const double value = X[i*cols + tid];
        //printf("Update read: %f\n", value);
        if (std::abs(value) > 1e-15) {
            sum += value;
            X[i*cols + tid] = 0;
            count += 1;
        }
    }
    //printf("sum: %f, count: %f\n", sum, count);
    //if (count > 0.5) printf("total update: %f\n", sum/count);
    //printf("position: %d, x before: %f, ", tid, x[tid]);
    if (count > 0.5) x[tid] += sum/count;
    //printf("x now: %f\n ", x[tid]);
  }
}


KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned rows,
    const unsigned cols, const unsigned nnz,
    const unsigned max_iterations, const double precision) {



  const unsigned L = 5000;  // we check for convergence every L steps
  bool converged = false;
  assert(rows == cols);
  const unsigned num_threads = cols;

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

  // move b to device
  double *d_b;
  cudaMalloc((void **)&d_b, cols * sizeof(double));
  cudaMemcpy(d_b, h_b, cols * sizeof(double), cudaMemcpyHostToDevice);

  //move X to device
  double *d_X;
  cudaMalloc((void **)&d_X, cols*cols*sizeof(double));
  cudaMemset((void**)d_X, 0, cols*cols*sizeof(double));

  // solve LSE
  for (int iter = 0; iter < max_iterations; iter++){
    step<<<1, num_threads>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x, d_X);
    update<<<1, num_threads>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x, d_X);
    
    //calculate residual every L iterations
    if (iter % L == 0 and iter > 0){
      cudaMemcpy(h_x, d_x, cols * sizeof(double), cudaMemcpyDeviceToHost);
      double residual = 0.0;

      for (unsigned i = 0; i < rows; i++) {
        double dot_product = 0.0;
        for (unsigned j = h_A_outer[i]; j < h_A_outer[i + 1]; j++) {
          dot_product += h_A_values[j] * h_x[h_A_inner[j]];
        }
        residual += (dot_product - h_b[i]) * (dot_product - h_b[i]);
      }
      residual = sqrt(residual);

      printf("Residual: %f\n", residual);

      if (residual < precision) {
        converged = true;
        break;
      }
    }
  }

  // free memory
  cudaFree(d_x);
  cudaFree(d_X);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);

  // check for convergence
  if (converged) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::OutOfIterations;
}