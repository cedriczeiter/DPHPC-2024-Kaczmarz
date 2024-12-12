#ifndef CARP_UTILS_HPP
#define CARP_UTILS_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

#define L_RESIDUAL 50
#define ROWS_PER_THREAD 100
#define THREADS_PER_BLOCK 512
#define RUNS_PER_THREAD 50

void add_gpu(const float* d_a, const float* d_b, float* d_output,
             const float factor, const unsigned dim);

void dcswp(const unsigned *d_affected, const int *d_A_outer, const int *d_A_inner, const float *d_A_values,
           const float *d_b, const unsigned dim, const float *d_sq_norms,
           const float *d_x, const float relaxation,
           const unsigned total_threads, float *d_output,
           float *d_intermediate, const unsigned blocks,
           curandState *state);

void copy_gpu(const float* d_from, float* d_to, const unsigned dim);
float dot_product_gpu(const float* d_a, const float* d_b, float* d_to,
                       const unsigned dim);

float get_residual(float *h_x, const float *h_b, const float *d_x,
                    const int *h_A_outer, const int *h_A_inner,
                    const float *h_A_values, const unsigned dim);

void setup_curand_gpu(curandState *state, const unsigned long seed, const unsigned total_threads);

#define CUDA_SAFE_CALL(call)                                                 \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, \
              __LINE__, cudaGetErrorString(err));                            \
      exit(-1);                                                              \
    }                                                                        \
  } while (0)

#define CUDA_SAFE_KERNEL_LAUNCH(kernel, ...)                             \
  do {                                                                   \
    kernel<<<__VA_ARGS__>>>();                                           \
    cudaError_t err = cudaGetLastError();                                \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr,                                                    \
              "CUDA kernel launch error in file '%s' in line %i: %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString(err));              \
      exit(-1);                                                          \
    }                                                                    \
    CUDA_SAFE_CALL(cudaDeviceSynchronize());                             \
  } while (0)

#endif  // CARP_UTILS_HPP