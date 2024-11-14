#include <unistd.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "asynchronous.hpp"
#include "common.hpp"

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads) {
  const unsigned rows = lse.row_count();
  // const unsigned cols = lse.column_count();

  assert(num_threads <=
         rows);  // necessary for allowing each thread to have local rows

  const unsigned L = 500;  // we check for convergence every L steps
  bool converged = false;

  const unsigned runs_per_thread = 15;

  // squared norms of rows of A (so that we don't need to recompute them in each
  // iteration
  std::vector<double> h_sq_norms(rows);
  for (unsigned i = 0; i < rows; i++) {
    h_sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
    if (h_sq_norms[i] < 1e-7) return KaczmarzSolverStatus::ZeroNormRow;
  }
  // allocate move squared norms on device
  double *d_sq_norms;
  cudaCheckError(cudaMalloc(&d_sq_norms, rows * sizeof(double)));
  cudaCheckError(cudaMemcpy(h_sq_norms.data(), d_sq_norms,
                            rows * sizeof(double), cudaMemcpyHostToDevice));

  // each thread chooses randomly from own set of rows
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
  }
  /*// move local rows on device
  unsigned *d_local_rows;
  cudaCheckError(cudaMalloc(&d_local_rows, num_threads * sizeof(unsigned)));
  cudaCheckError(cudaMemcpy(h_local_rows.data(), d_local_rows,
                            num_threads * sizeof(unsigned),
                            cudaMemcpyHostToDevice));*/

  // move x to device
  double *d_x;
  cudaCheckError(cudaMalloc(&d_x, x.size() * sizeof(double)));
  cudaCheckError(cudaMemcpy(x.data(), d_x, x.size() * sizeof(double),
                            cudaMemcpyHostToDevice));

  //create convergence flag and move to device
  bool h_converged;
  bool *d_converged;
  cudaMalloc(&d_converged, sizeof(bool));

  //solve LSE
  solve_async<<<max_blocks, block>>>(lse, d_sq_norms, d_x, max_iterations, runs_before_sync, d_converged);

  //copy back x and convergence
  cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);
  cudaMemcpy(x.data(), d_x, x.size()*sizeof(double), cudaMemcpyDeviceToHost);

  //free memory
  cudaFree(d_converged);
  cudaFree(d_x);
  cudaFree(d_sq_norms);

  //check for convergence
  if (h_converged) return KaczmarzSolverStatus::Converged;
  return KaczmarzSolverStatus::OutOfIterations;
}

__global__ void solve_async(const SparseLinearSystem &lse, const double *sq_norms, double *x, const unsigned max_iterations, const unsigned runs_before_sync, bool *converged){
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(21, tid, 0, &state);

  auto A = lse.A();
  auto b = lse.b();
  const unsigned rows = A.rows();

  for (unsigned iter = 0; iter < max_iterations; iter++){
    for (unsigned i = 0; i < runs_before_sync; i++){
      //get random row
      unsigned k = curand(&state) % rows;

      //compute dot product row * x
      double dot_product = 0.;
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it){
        dot_product += it.value() * *(x + it.col());
      }

      const double update_coeff = (b[k] - dot_product) / sq_norms[k];

      //update x
      for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, k); it; ++it){
        const double update = update_coeff * it.value();
        atomicAdd(x + it.col(), update);
      }
    }
    //sync all threads to guarantee convergence
    __syncthreads();
    if (iter % L == 0 && iter > 0 && thread_num == 0) {
            double residual = 0.0;
            for (unsigned i = 0; i < rows; i++) {
                double dot_product = 0.0;
                for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, i); it; ++it) {
                    dot_product += it.value() * x[it.col()];
                }
                residual += (dot_product - b[i]) * (dot_product - b[i]);
            }
            residual = sqrt(residual);

            if (residual < precision) {
                *converged = true;
                break;
            }
        }
    __syncthreads();
    if (*converged) break;
  }
}

/*#pragma omp parallel
  {
    const auto A = lse.A();
    const auto b = lse.b();
    std::mt19937 rng(21);

    const unsigned thread_num = omp_get_thread_num();
    const unsigned local_rows_size = local_rows[thread_num].size();
    std::uniform_int_distribution<> distr(0, local_rows_size - 1);

    // Loop performed by all threads
    for (unsigned iter = 0; iter < max_iterations; iter++) {
      for (unsigned i = 0; i < runs_per_thread; i++) {
        // Randomly select a row
        unsigned k = local_rows[thread_num].at(distr(rng));

        double dot_product = 0.0;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A,
                                                                            k);
             it; ++it) {
          double x_value;
#pragma omp atomic read
          x_value = x[it.col()];
          dot_product += it.value() * x_value;
        }

        double update_coeff = ((b[k] - dot_product) / sq_norms[k]);
        //  update
        for (SparseMatrix::InnerIterator it(A, k); it; ++it) {
          const double update = update_coeff * it.value();
#pragma omp atomic update
          x[it.col()] += update;
        }
      }
// we synchronize all threads, so we comply with the convergence
// conditions of the asynchronous algorithms
#pragma omp barrier

      // stopping criterion
      if (thread_num == 0 && iter % L == 0 &&
          iter > 0) {  // Check every L iterations
        double residual = (A * x - b).norm();
        if (residual < precision) {
#pragma omp atomic write
          converged = true;
        }
        // std::cout << residual << std::endl;
      }
      if (converged) {
#pragma omp cancel parallel
      }

#pragma omp cancellation point parallel
    }
  }
  if (converged) return KaczmarzSolverStatus::Converged;

  return KaczmarzSolverStatus::OutOfIterations;
}*/