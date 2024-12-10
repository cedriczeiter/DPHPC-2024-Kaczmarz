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
__global__ void kswp(const int *A_outer, const int *A_inner,
                     const double *A_values_shared, const double *b_local,
                     const unsigned dim, const double *sq_norms_local,
                     const double *x, const unsigned rows_per_thread,
                     const double relaxation, double *output,
                     const int *affected, bool forward,
                     const unsigned max_nnz_in_row) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid * rows_per_thread < dim)  // only if thread has assigned rows (dim)
  {
    // perform sweep
      switch (forward) {
        case true:
          for (unsigned k = 0; k < rows_per_thread && (tid*rows_per_thread + k) < dim; k++) {

            const unsigned row = tid * rows_per_thread + k;
            // compute dot product row * x
            double dot_product = 0.;

            const int a_outer_row = A_outer[row];
            const int a_outer_row_next = A_outer[row + 1];
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              const double x_value = x[A_inner[i]];
              dot_product += A_values_shared[i] * x_value;
            }

            // calculate update
            const double update_coeff =
                relaxation * ((b_local[row] - dot_product) /
                              (sq_norms_local[row] * (double)max_nnz_in_row));
            // printf("sq_norm: %f, update: %f\n", sq_norms_local[row],
            // update_coeff);
            //  save update for output
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              assert(affected[A_inner[i]] != 0);
              atomicAdd(&output[A_inner[i]], update_coeff * A_values_shared[i]);
            }
          }
          break;
        case false:
          for (int k = 0; k < rows_per_thread && (tid*rows_per_thread + k) < dim; k++) {
            unsigned row = ((tid+1) * rows_per_thread) - k - 1; //this is added to allow the algorithm to work even if the rows_per_thread does not cleanly divide the dimension of the matrix
            if (row >= dim){
              row -= rows_per_thread - (dim % rows_per_thread);
            }
            assert(row < dim);

            // compute dot product row * x
            double dot_product = 0.;
            const int a_outer_row = A_outer[row];
            const int a_outer_row_next = A_outer[row + 1];
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              const double x_value = x[A_inner[i]];
              dot_product += A_values_shared[i] * x_value;
            }
            // calculate update
            const double update_coeff =
                relaxation * ((b_local[row] - dot_product) /
                              (sq_norms_local[row] * (double)max_nnz_in_row));
            // save update for output
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              atomicAdd(&output[A_inner[i]],
                        /*(1. /  (double)affected[A_inner[i]])*/ update_coeff *
                            A_values_shared[i]);
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
void dcswp(const int *d_A_outer, const int *d_A_inner, const double *d_A_values,
           const double *d_b, const unsigned dim, const double *d_sq_norms,
           const double *d_x, const double relaxation, const int *d_affected,
           const unsigned total_threads, double *d_output,
           double *d_intermediate, const unsigned blocks,
           const unsigned max_nnz_in_row) {
  // copy x vector to output vector
  copy_gpu(d_x, d_intermediate, dim);
  // perform step forward
  kswp<<<blocks, THREADS_PER_BLOCK>>>(d_A_outer, d_A_inner, d_A_values, d_b,
                                      dim, d_sq_norms, d_x, ROWS_PER_THREAD,
                                      relaxation, d_intermediate, d_affected,
                                      true, max_nnz_in_row);

  auto res = cudaDeviceSynchronize();
  assert(res == 0);

  // copy intermediate vector over to output vector
  copy_gpu(d_intermediate, d_output, dim);
  // perform step backward
  kswp<<<blocks, THREADS_PER_BLOCK>>>(
      d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_intermediate,
      ROWS_PER_THREAD, relaxation, d_output, d_affected, false, max_nnz_in_row);

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
