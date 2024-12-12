#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <set>

#include "carp_utils.hpp"
#include "common.hpp"

// IMPORTANT: ONLY WORKS ON SQUARE MATRICES ATM AND IF ROWS_PER_THREAD DIVIDES
// TOTAL ROWS(dim)

KaczmarzSolverStatus invoke_carp_solver_gpu(
    const int *h_A_outer, const int *h_A_inner, const float *h_A_values,
    const float *h_b, float *h_x, float *h_sq_norms, const unsigned dim,
    const unsigned nnz, const unsigned max_iterations, const float precision,
    const unsigned max_nnz_in_row, const float b_norm, int &nr_of_steps,
    const float relaxation) {
  // define some variables
  bool converged = false;
  const unsigned total_threads = (dim + ROWS_PER_THREAD - 1) / ROWS_PER_THREAD;

  // allocate move squared norms on device
  float *d_sq_norms;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_sq_norms, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_sq_norms, h_sq_norms, dim * sizeof(float),
                            cudaMemcpyHostToDevice));

  // move x to device
  float *d_x;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, dim * sizeof(float)));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_x, h_x, dim * sizeof(float), cudaMemcpyHostToDevice));

  // move A to device
  int *d_A_outer;
  int *d_A_inner;
  float *d_A_values;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_outer, (dim + 1) * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_inner, nnz * sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_A_values, nnz * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_outer, h_A_outer, (dim + 1) * sizeof(int),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_inner, h_A_inner, nnz * sizeof(int),
                            cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(float),
                            cudaMemcpyHostToDevice));

  // move b to device
  float *d_b;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_b, dim * sizeof(float)));
  CUDA_SAFE_CALL(
      cudaMemcpy(d_b, h_b, dim * sizeof(float), cudaMemcpyHostToDevice));

  /*//we want to find a partitioning for our threads
  std::vector<std::vector<unsigned>> which_rows_to_thread((dim + ROWS_PER_THREAD-1)/ROWS_PER_THREAD);
  for (int i = 0; i < dim; i++){
    const unsigned thread = i/ROWS_PER_THREAD;
    which_rows_to_thread[thread].push_back(i);
  }*/

  //we find out which row is affected by which threads
  std::vector<std::set<unsigned>> which_rows_affected_by_thread(dim); //we create a set for each row, each thread which affects this row gets added to the set
  std::vector<std::set<unsigned>> which_entries_for_which_thread(total_threads);
  for (int i = 0; i < dim; i++){
    const unsigned thread = i/ROWS_PER_THREAD;
    for (int k = h_A_outer[i]; k < h_A_outer[i+1]; k++){
      which_rows_affected_by_thread[h_A_inner[k]].insert(thread);
      which_entries_for_which_thread[thread].insert(h_A_inner[k]);
    }
  }

  //copy number of threads affecting row over to array
  unsigned *h_affected = new unsigned[dim];
  for (int i = 0; i < dim; i++){
    h_affected[i] = (which_rows_affected_by_thread[i].size()); 
  }

  //copy over to device
  unsigned *d_affected;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_affected, dim * sizeof(unsigned)));
  CUDA_SAFE_CALL(cudaMemcpy(d_affected, h_affected, dim * sizeof(unsigned),
                            cudaMemcpyHostToDevice));

  delete[] h_affected;

  //get max number of entries per thread
  unsigned max_entries_per_thread = 0;
  for (int i = 0; i < total_threads; i++){
    max_entries_per_thread = std::max(max_entries_per_thread, (unsigned)which_entries_for_which_thread[i].size());
  }

  std::cout << "Max Entry per thread: " << max_entries_per_thread << std::endl;


  curandState *d_state;
  cudaMalloc((void **)&d_state, total_threads * sizeof(curandState)); 

  //init random number generators
  setup_curand_gpu(d_state, 13, total_threads);

  // move p, r, q and intermediate storage to device
  float *d_p;
  float *d_r;
  float *d_q;
  float *d_intermediate;
  float *d_intermediate_two;
  float *d_zero;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_p, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_r, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_q, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_intermediate, dim * sizeof(float)));
  CUDA_SAFE_CALL(
      cudaMalloc((void **)&d_intermediate_two, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_zero, dim * sizeof(float)));
  CUDA_SAFE_CALL(cudaMemset((void **)d_zero, 0, dim * sizeof(float)));

  // calculate nr of blocks and threads
  const int blocks =
      (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  // solve LSE:
  // init stuff
  float residual = 1.;  // init value, will be overwritten as soon as we check
                         // for convergence
  /*dcswp(d_affected, d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_x, relaxation, total_threads, d_intermediate, d_intermediate_two, blocks, max_nnz_in_row);
  add_gpu(d_intermediate, d_x, d_r, -1, dim);
  copy_gpu(d_r, d_p, dim);*/

  for (int iter = 0; iter < max_iterations; iter++) {
    // calculate residual every L_RESIDUAL iterations
    if (iter % L_RESIDUAL == 0) {
      residual =
          get_residual(h_x, h_b, d_x, h_A_outer, h_A_inner, h_A_values, dim);
      // debugging output
      std::cout << "Iteration: " << iter << " out of " << max_iterations
              << " , Residual/B_norm: " << residual / b_norm << std::endl;
      //std::cout << "Entry 0: " << h_x[0] << std::endl;
      // check for convergence
      if (residual / b_norm < precision) {
        converged = true;
        nr_of_steps = iter;
        break;  // stop all the iterations
      }
    }
    /*dcswp(d_affected, d_A_outer, d_A_inner, d_A_values, d_zero, dim, d_sq_norms, d_p,
          relaxation, total_threads, d_intermediate, d_intermediate_two, blocks,
          max_nnz_in_row);
    add_gpu(d_p, d_intermediate, d_q, -1, dim);
    const float sq_norm_r = dot_product_gpu(d_r, d_r, d_intermediate, dim);
    const float dot_p_q = dot_product_gpu(d_p, d_q, d_intermediate, dim);
    const float alpha = sq_norm_r/dot_p_q;
    add_gpu(d_x, d_p, d_x, alpha,  dim);
    add_gpu(d_r, d_q, d_r, -alpha, dim);
    const float beta = dot_product_gpu(d_r, d_r, d_intermediate, dim)/sq_norm_r;
    add_gpu(d_r, d_p, d_p, beta, dim);*/
    dcswp(d_affected, d_A_outer, d_A_inner, d_A_values, d_b, dim, d_sq_norms, d_x,
          relaxation, total_threads, d_intermediate, d_intermediate_two, blocks,
          d_state);
    copy_gpu(d_intermediate, d_x, dim);
  }

  // free memory
  cudaFree(d_x);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);
  cudaFree(d_p);
  cudaFree(d_r);
  cudaFree(d_q);
  cudaFree(d_intermediate);
  cudaFree(d_intermediate_two);
  cudaFree(d_zero);
  cudaFree(d_affected);
  // check for convergence
  if (converged) {
    return KaczmarzSolverStatus::Converged;
  } else {
    return KaczmarzSolverStatus::OutOfIterations;
  }
}