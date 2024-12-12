#include <cuda_runtime.h>
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

__global__ void setup_curand(curandState *state, const unsigned long seed, const unsigned total_threads){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < total_threads){
      curand_init(seed, id, 0, &state[id]);
    }
}

// The function kswp is the main kernel function of the CARP solver that
// performs calculations on the GPU
__global__ void kswp(const unsigned *affected, const int *A_outer, const int *A_inner,
                     const float *A_values_shared, const float *b_local,
                     const unsigned dim, const float *sq_norms_local, const unsigned rows_per_thread,
                     const float relaxation, float *output, bool forward, curandState *state) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid * rows_per_thread < dim)  // only if thread has assigned rows (dim)
  {
    // perform sweep
    curandState localState = state[tid];
          for (unsigned k = 0; k < RUNS_PER_THREAD; k++) {
            const int row = min(dim-1, (int)(ceil((curand_uniform(&localState)*(rows_per_thread + 1))) - 1) + tid*rows_per_thread);
            // compute dot product row * x
            float dot_product = 0.;

            const int a_outer_row = A_outer[row];
            const int a_outer_row_next = A_outer[row + 1];
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              const float x_value = output[A_inner[i]];
              dot_product += A_values_shared[i] * x_value;
            }

            // calculate update
            const float update_coeff =
                relaxation *
                ((b_local[row] - dot_product) / (sq_norms_local[row]));
            // printf("sq_norm: %f, update: %f\n", sq_norms_local[row],
            // update_coeff);
            //  save update for local x
            for (unsigned i = a_outer_row; i < a_outer_row_next; i++) {
              const float x_value = output[A_inner[i]];
              atomicExch(&output[A_inner[i]], x_value + update_coeff * A_values_shared[i]);
            }
      }
      state[tid] = localState;
    }
}

__global__ void add(const float *a, const float *b, float *output,
                    const float factor, const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = a[tid] + factor * b[tid];
    // printf("Adding: %lf, %lf, %lf\n", a[tid], b[tid], output[tid]);
  }
}

__global__ void copy(const float *from, float *to, const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    to[tid] = from[tid];
    // printf("Copying: %lf, %lf\n", from[tid], to[tid]);
  }
}

__global__ void set_zero(float *output, const unsigned dim){
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = 0;
    // printf("Copying: %lf, %lf\n", from[tid], to[tid]);
  }
}

__global__ void reduce(float *data, const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid+ (dim+1)/2 < dim) {
    data[tid] += data[tid + (dim+1)/2];
  }
}

__global__ void square_vector(const float *a, const float *b, float *output,
                              const unsigned dim) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < dim) {
    output[tid] = a[tid] * b[tid];
    // printf("Squaring: %lf, %lf, %lf\n", a[tid], b[tid], output[tid]);
  }
}

void add_gpu(const float *d_a, const float *d_b, float *d_output,
             const float factor, const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  add<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_output, factor, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

void setup_curand_gpu(curandState *state, const unsigned long seed, const unsigned total_threads){
  const int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  setup_curand<<<blocks, THREADS_PER_BLOCK>>>(state, seed, total_threads);
}

void copy_gpu(const float *d_from, float *d_to, const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  copy<<<blocks, THREADS_PER_BLOCK>>>(d_from, d_to, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

void set_zero_gpu(float *d_output,  const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  set_zero<<<blocks, THREADS_PER_BLOCK>>>(d_output, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

float dot_product_gpu(const float *d_a, const float *d_b, float *d_to,
                       const unsigned dim) {
  // Calculate the number of blocks needed
  const int blocks = (dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  square_vector<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_to, dim);
  auto res = cudaDeviceSynchronize();
  assert(res == 0);
  unsigned current_dim = dim;
  while (current_dim > 1){
    const unsigned current_blocks = ((current_dim+1)/2 + THREADS_PER_BLOCK - 1) /THREADS_PER_BLOCK;
    reduce<<<current_blocks, THREADS_PER_BLOCK>>>(d_to, current_dim);
    current_dim = (current_dim+1)/2;
  }
  float dot_product;
  CUDA_SAFE_CALL(cudaMemcpy(&dot_product, d_to, sizeof(float), cudaMemcpyDeviceToHost));
  return dot_product;
}

// Function to perform the sweep forward and backward (main function of the CARP
// solver)
void dcswp(const unsigned *d_affected, const int *d_A_outer, const int *d_A_inner, const float *d_A_values,
           const float *d_b, const unsigned dim, const float *d_sq_norms,
           const float *d_x, const float relaxation,
           const unsigned total_threads, float *d_output,
           float *d_intermediate, const unsigned blocks,
           curandState *state) {
  // first output will go into intermediate vector, therefore set to zeor
  copy_gpu(d_x, d_output, dim);
  // perform step forward
  kswp<<<blocks, THREADS_PER_BLOCK>>>(d_affected, d_A_outer, d_A_inner, d_A_values, d_b,
                                      dim, d_sq_norms, ROWS_PER_THREAD,
                                      relaxation, d_output, true, state);

  auto res = cudaDeviceSynchronize();
  assert(res == 0);
}

// copies x from device to host, and calculates residual
float get_residual(float *h_x, const float *h_b, const float *d_x,
                    const int *h_A_outer, const int *h_A_inner,
                    const float *h_A_values, const unsigned dim) {
  cudaMemcpy(h_x, d_x, dim * sizeof(float), cudaMemcpyDeviceToHost);
  float residual = 0.0;
  // Calulate residual
  for (unsigned i = 0; i < dim; i++) {
    float dot_product = 0.0;
    for (unsigned j = h_A_outer[i]; j < h_A_outer[i + 1]; j++) {
      dot_product += h_A_values[j] * h_x[h_A_inner[j]];
    }
    residual += (dot_product - h_b[i]) * (dot_product - h_b[i]);
  }
  residual = sqrt(residual);
  return residual;
}
