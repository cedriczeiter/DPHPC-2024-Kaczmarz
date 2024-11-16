#include <unistd.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "asynchronous.hpp"
#include "common.hpp"

__global__ void solve_async(const int* A_outerIndex, const int* A_innerIndex, const double* A_values, const double* b, const unsigned rows, const unsigned cols, const double *sq_norms, double *x, const unsigned max_iterations, const unsigned runs_before_sync, bool *converged, const unsigned L, const double precision, const unsigned num_threads){
  
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(13, tid, 0, &state);

  //we want each thread to operate on set of own rows (mutually almost exclusive)
  const unsigned n_own_rows = (rows/num_threads) + 1;
  const unsigned thread_offset = n_own_rows*tid;

  printf("B: \n");
  for (int i = 0; i < cols; i++){
    printf("%f\n", b[i]);
  }


  for (unsigned iter = 0; iter < max_iterations; iter++){
    for (unsigned inner_iter = 0; inner_iter < runs_before_sync; inner_iter++){

      //get random row
      const unsigned k = thread_offset + (curand(&state) % n_own_rows);

      if (k >= rows) continue;


      //compute dot product row * x
      double dot_product = 0.;
      for (unsigned i = A_outerIndex[k]; i < A_outerIndex[k+1]; i++){
        const double x_value = atomicAdd(&x[A_innerIndex[i]], 0.);
        dot_product += A_values[i] * x_value;
      }

      printf("k: %d\n", k);
      printf("dot product: %f\n", dot_product);

      printf("norm: %f\n", sq_norms[k]);

      printf("b_k: %f\n", b[k]);

      const double update_coeff = ((b[k] - dot_product) / sq_norms[k]);

      printf("Update Coeff: %f\n", update_coeff);

      //update x
      for (unsigned i = A_outerIndex[k]; i < A_outerIndex[k+1]; i++){
        const double update = update_coeff * A_values[i];
        atomicAdd(&x[A_innerIndex[i]], update);
      }
    }
    //sync all threads to guarantee convergence
    __syncthreads();
    if (iter % 10 == 0 && iter > 0 && tid == 0) {
            double residual = 0.0;
            for (unsigned i = 0; i < rows; i++) {
                double dot_product = 0.0;
                for (unsigned j = A_outerIndex[i]; j < A_outerIndex[i+1]; j++){
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

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads) {
  const unsigned rows = lse.row_count();
  const unsigned nnz = lse.A().nonZeros();
  const unsigned cols = lse.column_count();
  

  assert(num_threads <=
         rows);  // necessary for allowing each thread to have local rows

  const unsigned L = 500;  // we check for convergence every L steps
  const unsigned runs_before_sync = 30;
  bool converged = false;

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  std::vector<double> h_sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    h_sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    std::cout << "Row: " << i << " Norm: " << h_sq_norms[i] << std::endl;
    if (h_sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }
  // allocate move squared norms on device
  double *d_sq_norms;
  cudaMalloc((void **)&d_sq_norms, rows * sizeof(double));
  cudaMemcpy(d_sq_norms, h_sq_norms.data(),
                            rows * sizeof(double), cudaMemcpyHostToDevice);


  /*// each thread chooses randomly from own set of rows
  unsigned rows_per_thread = (unsigned)(rows / num_threads);
  std::vector<std::vector<unsigned> > h_local_rows(num_threads);
  for (unsigned i = 0; i < num_threads; i++) {
    for (unsigned j = rows_per_thread * i;
         j < rows && j < rows_per_thread * (i + 1); j++) {
      h_local_rows.at(i).push_back(j);
    }
  }
  for (unsigned j = rows_per_thread * num_threads; j < rows; j++) {
    h_local_rows.at(num_threads - 1).push_back(j);
  }*/
  /*// move local rows on device
  unsigned *d_local_rows;
  cudaCheckError(cudaMalloc(&d_local_rows, num_threads * sizeof(unsigned)));
  cudaCheckError(cudaMemcpy(h_local_rows.data(), d_local_rows,
                            num_threads * sizeof(unsigned),
                            cudaMemcpyHostToDevice));*/

  // move x to device
  double *d_x;
  cudaMalloc((void **)&d_x, x.size() * sizeof(double));
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(double),
                            cudaMemcpyHostToDevice);

  //create convergence flag and move to device
  bool *h_converged = new bool(false);
  bool *d_converged;
  cudaMalloc((void **)&d_converged, sizeof(bool));
  cudaMemcpy(d_converged, h_converged, sizeof(bool), cudaMemcpyHostToDevice);


  //move A to device
  int *d_A_outer;
  int *d_A_inner;
  double *d_A_values;
  cudaMalloc((void **)&d_A_outer, rows*sizeof(int));
  cudaMalloc((void **)&d_A_inner, nnz*sizeof(int));
  cudaMalloc((void **)&d_A_values, nnz*sizeof(double));
  cudaMemcpy(d_A_outer,  lse.A().outerIndexPtr(), rows*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_A_inner,  lse.A().innerIndexPtr(), nnz*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_A_values,  lse.A().valuePtr(), nnz*sizeof(double), cudaMemcpyHostToDevice);


  //move b to device
  double *d_b;
  cudaMalloc((void **)&d_b, x.size()*sizeof(double));
  cudaMemcpy( d_b,  lse.b().data(), x.size()*sizeof(double), cudaMemcpyHostToDevice);
  //solve LSE
  std::cout << "Calling kernel\n";
  solve_async<<<1, num_threads>>>(d_A_outer, d_A_inner, d_A_values, d_b, rows, cols, d_sq_norms, d_x, 1000, runs_before_sync, d_converged, L, precision, num_threads);
  cudaDeviceSynchronize();
  std::cout << "Kernel done\n";
  //copy back x and convergence
  cudaMemcpy(h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy( x.data(),  d_x, x.size()*sizeof(double), cudaMemcpyDeviceToHost);

  /*for (int i = 0; i < x.size(); i++){
    std::cout << (lse.b())[i] << " ";
  }
  std::cout << std::endl;*/

  //for testing
  std::cout << "Normal: " << std::endl;
  for (int i = 0; i < rows; i++){
    std::cout << (lse.A().row(i)).dot(x) << std::endl;
  }
  std::cout << "Using pointers: " << std::endl;
  auto outer = lse.A().outerIndexPtr();
  auto values = lse.A().valuePtr();
  auto inner = lse.A().innerIndexPtr();
  for (int j = 0; j < rows; j++){
    double dot = 0;
    for (int i = outer[j]; i < outer[j+1]; i++){
      dot += x[inner[i]]*values[i];
    }
    std::cout << dot << std::endl;
  }

  //free memory
  cudaFree(d_converged);
  cudaFree(d_x);
  cudaFree(d_sq_norms);
  cudaFree(d_A_outer);
  cudaFree(d_A_inner);
  cudaFree(d_A_values);
  cudaFree(d_b);

  //check for convergence
  if (*h_converged) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::OutOfIterations;
}