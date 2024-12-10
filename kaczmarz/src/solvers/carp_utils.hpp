#ifndef CARP_UTILS_HPP
#define CARP_UTILS_HPP

#include <cuda_runtime.h>
#include <stdio.h>

#define L_RESIDUAL 2000
#define ROWS_PER_THREAD 1 //IMPORTANT: THIS CAN NOT BE LARGER THAN THE DIMENSION OF THE MATRIX!!
#define THREADS_PER_BLOCK 1024

void add_gpu(const double* d_a, const double* d_b, double* d_output,
             const double factor, const unsigned dim);

void dcswp(
           const double* d_b, const unsigned dim, 
           const double* d_x, const double relaxation,
           const unsigned total_threads, double* d_output,
           double* d_intermediate, const unsigned blocks,
           const unsigned max_nnz_in_row, const int* const* d_all_padded_inner, const double* const* d_all_padded_values);

void copy_gpu(const double* d_from, double* d_to, const unsigned dim);
double dot_product_gpu(const double* d_a, const double* d_b, double* d_to,
                       const unsigned dim);

double get_residual(double* h_x, const double* h_b, const double* d_x,
                    const int* h__A_outer, const int* h_A_inner,
                    const double* h_A_values, const unsigned dim);

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