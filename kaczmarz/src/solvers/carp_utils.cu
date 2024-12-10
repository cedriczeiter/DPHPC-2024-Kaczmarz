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

#include "carp_cuda.hpp"
#include "carp_utils.hpp"
#include "common.hpp"

// The function kswp is the main kernel function of the CARP solver that
// performs calculations on the GPU
__global__ void kswp(const double *b_local,
                     const unsigned dim,
                     const double *x, const unsigned rows_per_thread,
                     const double relaxation, double *output, bool forward, const int* const* d_all_padded_inner, const double* const* d_all_padded_values) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid * rows_per_thread < dim)  // only if thread has assigned rows (dim)
  {
    // perform sweep
    // INFO: for the carp-cg algorithm, only one run per thread should be used
    for (unsigned local_iter = 0; local_iter < LOCAL_RUNS_PER_THREAD;
         local_iter++) {
      
      switch (forward) {
        case true:
          for (unsigned k = 0; k < rows_per_thread; k++) {

            const unsigned row = tid * rows_per_thread + k;

            const int* inner = d_all_padded_inner[row];
            const double* values = d_all_padded_values[row];
            const int values_in_row = inner[0];

            assert(rows_per_thread == 1);
            assert(k == 0);
            // compute dot product row * x
            double dot_product = 0.;

            for (unsigned i = 0; i < values_in_row; i++) {
              const double x_value = x[inner[i+1]];
              dot_product += values[i+1] * x_value;
            }

            // calculate update
            const double update_coeff =
                relaxation *
                ((b_local[row] - dot_product) / values[0]);
            // printf("sq_norm: %f, update: %f\n", sq_norms_local[row],
            // update_coeff);
            //  save update for output
            for (unsigned i = 0; i < values_in_row; i++) {
              atomicAdd(&output[inner[i+1]], update_coeff * values[i+1]);
            }
          }
          break;
        case false:
          for (int k = rows_per_thread - 1; k >= 0; k--) {
            const unsigned row = tid * rows_per_thread + k;

            const int* inner = d_all_padded_inner[row];
            const double* values = d_all_padded_values[row];
            const int values_in_row = inner[0];

            // compute dot product row * x
            double dot_product = 0.;

            for (unsigned i = 0; i < values_in_row; i++) {
              const double x_value = x[inner[i+1]];
              dot_product += values[i+1] * x_value;
            }
            // calculate update
            const double update_coeff =
                relaxation *
                ((b_local[row] - dot_product) / values[0]);
            // save update for output
            for (unsigned i = 0; i < values_in_row; i++) {
              atomicAdd(&output[inner[i+1]],update_coeff * values[i+1]);
            }
          }
      }
    }
  }
}

__global__ void add(const double *a, const double *b, double *output,
                    const double factor, const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = a[tid] + factor * b[tid];
    // printf("Adding: %lf, %lf, %lf\n", a[tid], b[tid], output[tid]);
  }
}

__global__ void copy(const double *from, double *to, const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    to[tid] = from[tid];
    // printf("Copying: %lf, %lf\n", from[tid], to[tid]);
  }
}

__global__ void square_vector(const double *a, const double *b, double *output,
                              const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = a[tid] * b[tid];
    // printf("Squaring: %lf, %lf, %lf\n", a[tid], b[tid], output[tid]);
  }
}

void add_gpu(const double *d_a, const double *d_b, double *d_output,
             const double factor, const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_output, factor, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

void copy_gpu(const double *d_from, double *d_to, const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  copy<<<blocks, THREADS_PER_BLOCK>>>(d_from, d_to, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

double dot_product_gpu(const double *d_a, const double *d_b, double *d_to,
                       const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  square_vector<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_to, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);

  double h_intermediate[dim];
  cudaMemcpy(h_intermediate, d_to, dim * sizeof(double),
             cudaMemcpyDeviceToHost);
  double dot = 0;
  for (unsigned i = 0; i < dim; i++) {
    double value = h_intermediate[i];
    dot += value;
  }
  return dot;
}

// Function to perform the sweep forward and backward (main function of the CARP
// solver)
void dcswp(
           const double* d_b, const unsigned dim,
           const double* d_x, const double relaxation,
           const unsigned total_threads, double* d_output,
           double* d_intermediate, const unsigned blocks,
           const unsigned max_nnz_in_row, const int* const* d_all_padded_inner, const double* const* d_all_padded_values) {
  // copy x vector to output vector
  copy_gpu(d_x, d_intermediate, dim);
  // perform step forward
  kswp<<<blocks, THREADS_PER_BLOCK>>>( d_b,
                                      dim, d_x, ROWS_PER_THREAD,
                                      relaxation, d_intermediate, true, d_all_padded_inner, d_all_padded_values);

  auto res = cudaDeviceSynchronize();
  assert(res == 0);

  // copy intermediate vector over to output vector
  copy_gpu(d_intermediate, d_output, dim);
  // perform step backward
  kswp<<<blocks, THREADS_PER_BLOCK>>>(
       d_b, dim, d_intermediate,
      ROWS_PER_THREAD, relaxation, d_output, false, d_all_padded_inner, d_all_padded_values);

  res = cudaDeviceSynchronize();
  assert(res == 0);
}

// copies x from device to host, and calculates residual
double get_residual(double *h_x, const double *h_b, const double *d_x,
                    const int *h_A_outer, const int *h_A_inner,
                    const double *h_A_values, const unsigned dim) {
  cudaMemcpy(h_x, d_x, dim * sizeof(double), cudaMemcpyDeviceToHost);
  double residual = 0.0;
  // Calulate residual
  for (unsigned i = 0; i < dim; i++) {
    double dot_product = 0.0;
    for (unsigned j = h_A_outer[i]; j < h_A_outer[i + 1]; j++) {
      dot_product += h_A_values[j] * h_x[h_A_inner[j]];
    }
    residual += (dot_product - h_b[i]) * (dot_product - h_b[i]);
  }
  residual = sqrt(residual);
  return residual;
}
