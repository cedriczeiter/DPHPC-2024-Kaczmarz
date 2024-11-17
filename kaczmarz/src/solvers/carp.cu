#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <cstring>
#include <cassert>


#include "common.hpp"

#define LAMBDA 1.5

//IMPORTANT: ONLY WORKS ON SQUARE MATRICES ATM

__global__ void step(const int *A_outerIndex, const int *A_innerIndex,
                            const double *A_values, const double *b,
                            const unsigned rows, const unsigned cols,
                            const double *sq_norms, double *x, double *X, const unsigned rows_per_thread) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  //printf("Before allocating\n");
  extern __shared__ int data[];
  int *A_outer = data;
  int *A_inner = &data[rows*rows];
  double* A_values_shared = (double*)&A_inner[rows*rows];
  double *X_local = (double*)&A_values_shared[rows*rows];
  //printf("After allocating\n");

  if (tid*rows_per_thread < rows){
    //copy over A to shared memory
    for (unsigned k = 0; k <= rows_per_thread; k++){
      A_outer[tid*rows_per_thread + k] = A_outerIndex[tid*rows_per_thread + k];
    }
    //printf("A_outer\n");
    for (unsigned k = A_outer[tid*rows_per_thread]; k < A_outer[(tid+1)*rows_per_thread]; k++){
      A_inner[k] = A_innerIndex[k];
      A_values_shared[k] = A_values[k];
    }
    //copy over X
    for (unsigned k = A_outer[tid*rows_per_thread]; k < A_outer[(tid+1)*rows_per_thread]; k++){
      X_local[tid*rows + A_inner[k]] = x[A_inner[k]];
      //printf("X at %d: %f\n", A_inner[k], x[A_inner[k]]);
    }
    //printf("A_inner and values\n");
    //perform one update step for assigned row
    for (unsigned local_iter = 0; local_iter < 10; local_iter++){
      for (unsigned k = 0; k < rows_per_thread; k++){
        //printf("Thread: %d, Assigned row: %d\n", tid, tid*rows_per_thread+k);
        // compute dot product row * x
        double dot_product = 0.;
        for (unsigned i = A_outer[tid*rows_per_thread + k]; i < A_outer[tid*rows_per_thread + k + 1]; i++) {
            const double x_value = X_local[tid*rows+A_inner[i]];
            dot_product += A_values_shared[i] * x_value;
        }
        //calculate update
        const double update_coeff = ((b[tid*rows_per_thread + k] - dot_product) / sq_norms[tid*rows_per_thread + k]);
        // save update for x in global matrix, will be used in average step
        for (unsigned i = A_outer[tid*rows_per_thread + k]; i < A_outer[tid*rows_per_thread + k + 1]; i++) {
            const double update = update_coeff * A_values_shared[i];
            X_local[tid*rows + A_inner[i]] += 1.5*update;
            //printf("Update: %f\n", update);
        }
      }
    }

    //set all values back in global matrix for averaging step
    for (int k = 0; k < rows_per_thread; k++){
      for (unsigned i = A_outer[tid*rows_per_thread + k]; i < A_outer[tid*rows_per_thread + k + 1]; i++) {
        X[(tid*rows_per_thread+k)*rows + A_inner[i]] = X_local[tid*rows + A_inner[i]];
      }
    }
  }
}

__global__ void update(const int *A_outerIndex, const int *A_innerIndex,
                            const double *A_values, const double *b,
                            const unsigned rows, const unsigned cols,
                            const double *sq_norms, double *x, double *X, int *affected, const unsigned rows_per_thread) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid*rows_per_thread < cols){
    for (unsigned k = 0; k < rows_per_thread; k++){
      //sum up updates for assigned entry
      double sum = 0;
      int counter = 0;
      while (true){
          int affecting_thread = affected[(tid*rows_per_thread + k)*rows + counter];
          if (affecting_thread < 0) break;
          counter++;
          const double value = X[affecting_thread*rows + tid*rows_per_thread + k];
          //printf("Update read: %f\n", value);
          sum += value;
      }
      //printf("thread: %d, row: %d, sum: %f, count: %d\n", tid, tid*rows_per_thread + k, sum, counter);
      //if (count > 0.5) printf("total update: %f\n", sum/count);
      //printf("position: %d, x before: %f, ", tid, x[tid]);
      if (counter > 0) x[tid*rows_per_thread + k] = sum/(double)counter;
      //printf("x now: %f\n ", x[tid]);
    }
  }
}


KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const double *h_A_values,
    const double *h_b, double *h_x, double *h_sq_norms, const unsigned rows,
    const unsigned cols, const unsigned nnz,
    const unsigned max_iterations, const double precision) {



  const unsigned L = 10000;  // we check for convergence every L steps
  bool converged = false;
  assert(rows == cols);

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
  //std::cout << "Before affecting" << std::endl;
  //calculate indices which affect other rows, and move to device
  std::vector<std::vector<unsigned>> affects(rows); //coding: affects[j][i]: the x at position j is affected by thread i
  for (unsigned k = 0; k < rows; k++){
    for (unsigned i = h_A_outer[k]; i < h_A_outer[k+1]; i++){
      unsigned row = k;
      unsigned col = h_A_inner[i];
      affects.at(row).push_back(col);
    }
  }
  //std::cout << "Affecting middle" << std::endl;
  int* h_affected = new int[rows*rows];
  //std::cout << "Before memsetting" << std::endl;
  std::memset(h_affected, -1, rows*rows*sizeof(int));
  for (int k = 0; k < affects.size(); k++){
    for (int i = 0; i < affects.at(k).size(); i++){
      //std::cout << "K: " << k << " I: " << i << " Outer size: " << affects.size() << " Inner size: " << affects.at(k).size() << std::endl;
      h_affected[k*rows + i] = affects.at(k).at(i);
    }
  }
  //std::cout << "After affecting" << std::endl;
  int *d_affected;
  cudaMalloc((void **)&d_affected, rows*rows*sizeof(int));
  cudaMemcpy(d_affected, h_affected, rows*rows*sizeof(int), cudaMemcpyHostToDevice);


  // move b to device
  double *d_b;
  cudaMalloc((void **)&d_b, cols * sizeof(double));
  cudaMemcpy(d_b, h_b, cols * sizeof(double), cudaMemcpyHostToDevice);

  //move X to device
  double *d_X;
  cudaMalloc((void **)&d_X, (cols+1)*cols*sizeof(double));
  cudaMemset((void**)d_X, 0, (cols+1)*cols*sizeof(double));

  //calculate nr of blocks and threads
  const int threads_per_block = 512;
  const int blocks = (rows + threads_per_block - 1)/threads_per_block;

  //std::cout << "Blocks: " << blocks << " . Threads per block: " << threads_per_block << std::endl;

  // solve LSE
  for (int iter = 0; iter < max_iterations; iter++){
    step<<<blocks, threads_per_block, 10000>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x, d_X, 10);
        auto res = cudaDeviceSynchronize();
        assert(res == 0);
    update<<<blocks, threads_per_block>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x, d_X, d_affected, 10);
        res = cudaDeviceSynchronize();
        assert(res == 0);
    
    //calculate residual every L iterations
    if (iter % 1 == 0 and iter > 0){
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
  cudaFree(d_affected);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);

  // check for convergence
  if (converged) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::OutOfIterations;
}