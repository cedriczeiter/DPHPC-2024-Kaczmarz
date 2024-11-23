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
#include "carp_cuda.hpp"
#include "carp_utils.hpp"



__global__ void kswp(const int *A_outer, const int *A_inner,
                     const double *A_values_shared, const double *b_local,
                     const unsigned dim,
                     const double *sq_norms_local, const double *x,
                     const unsigned rows_per_thread, const double relaxation, double* output,const int *affected, bool forward) {

  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  const unsigned rows = dim;
  const unsigned cols = dim;

  if (tid * rows_per_thread < rows)  // only if thread has assigned rows
  {

    // perform sweep
    //INFO: for the carp-cg algorithm, only one run per thread should be used
    for (unsigned local_iter = 0; local_iter < LOCAL_RUNS_PER_THREAD;
         local_iter++) {


      //first forward forward...
    if (forward){
      for (unsigned k = 0; k < rows_per_thread; k++) {
        const unsigned row = tid * rows_per_thread + k;
          // compute dot product row * x
          double dot_product = 0.;
          for (unsigned i = A_outer[row];
              i < A_outer[row + 1]; i++) {
            const double x_value = output[A_inner[i]];
            dot_product += A_values_shared[i] * x_value;
          }
          // calculate update
          const double update_coeff =
              relaxation * ((b_local[row] - dot_product) /
                            sq_norms_local[row]);
          // save update for output
          for (unsigned i = A_outer[row];
              i < A_outer[row + 1]; i++) {
                assert(affected[i] != 0);
            atomicAdd(&output[A_inner[i]], (1./(double)affected[A_inner[i]]) * update_coeff * A_values_shared[i]);
          }
        }
    }
    else{
      //then backward
        for (int k = rows_per_thread-1; k >= 0; k--) {
          const unsigned row = tid * rows_per_thread + k;
          // compute dot product row * x
          double dot_product = 0.;
          for (unsigned i = A_outer[row];
              i < A_outer[row + 1]; i++) {
            const double x_value = output[A_inner[i]];
            dot_product += A_values_shared[i] * x_value;
          }
          // calculate update
          const double update_coeff =
              relaxation * ((b_local[row] - dot_product) /
                            sq_norms_local[row]);
          // save update for output
          for (unsigned i = A_outer[row];
              i < A_outer[row + 1]; i++) {
           atomicAdd(&output[A_inner[i]], (1./(double)affected[A_inner[i]]) * update_coeff * A_values_shared[i]);
          }
        }
    }
    }
  }
}

__global__ void add(const double* a, const double* b, double* output, const double factor, const unsigned dim){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim){
        output[tid] = a[tid] + factor*b[tid];
    }
}

__global__ void copy(const double*from, double* to, const unsigned dim){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim){
        to[tid] = from[tid];
    }
}

__global__ void square_vector(const double *a, const double *b, double* output, const unsigned dim){
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim){
        output[tid] = a[tid]*b[tid];
    }
}

void add_gpu(const double* d_a, const double* d_b, double* d_output, const double factor, const unsigned dim){
    //calculate how many threads needed
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_output, factor, dim);
    auto res = cudaDeviceSynchronize();
    assert(res == 0);
}

void copy_gpu(const double* d_from, double* d_to, const unsigned dim){
    //calculate how many threads needed
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    copy<<<blocks, threadsPerBlock>>>(d_from, d_to, dim);
    auto res = cudaDeviceSynchronize();
    assert(res == 0);
}

double dot_product_gpu(const double* d_a, const double* d_b, double *d_to, const unsigned dim){
    //calculate how many threads needed
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed
    int blocks = (dim + threadsPerBlock - 1) / threadsPerBlock;
    square_vector<<<blocks, threadsPerBlock>>>(d_a, d_b, d_to, dim);
    auto res = cudaDeviceSynchronize();
    assert(res == 0);

    double h_intermediate[dim];
    cudaMemcpy(h_intermediate, d_to, dim * sizeof(double), cudaMemcpyDeviceToHost);
    double dot = 0;
    for (unsigned i = 0; i < dim; i++){
        double value = h_intermediate[i];
        dot += value;
    }
    return dot;
}



void dcswp(const int *d_A_outer, const int *d_A_inner,
                     const double *d_A_values, const double *d_b,
                     const unsigned dim,
                     const double *d_sq_norms, const double *d_x,
                     const double relaxation, const int *d_affected, const unsigned total_threads, double* d_output, const unsigned blocks){

  //copy x vector to output vector
  copy_gpu(d_x, d_output, dim);
  // perform step forward
    kswp<<<blocks, THREADS_PER_BLOCK>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_x,
        ROWS_PER_THREAD, relaxation, d_output, d_affected, true);
        // perform step backward
    kswp<<<blocks, THREADS_PER_BLOCK>>>(
        d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_x,
        ROWS_PER_THREAD, relaxation, d_output, d_affected, false);
}
