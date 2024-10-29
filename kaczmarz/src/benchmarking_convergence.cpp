#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "kaczmarz.hpp"
#include "kaczmarz_common.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "random_kaczmarz.hpp"

#define MAX_IT 1000000
#define BANDWIDTH 4
#define PRECISION 1e-10
#define TEST_DIM 16
#define NUM_ITERATIONS 1
#define CONVERGENCE_TRACKING_RATE 500

/// @brief Exports the convergence rates we got from the algorithms to a csv file.
/// @param filename A name of a csv file it should be exported to.
/// @param iterations A vector of iterations recorded for benchmarking.
/// @param times A vector of times recorded for benchmarking.
/// @param residuals A vector of residuals recorded for benchmarking
void export_convergence_to_csv(const std::string& filename,
                   const std::vector<int>& iterations,
                   const std::vector<double>& times,
                   const std::vector<double>& residuals) {
    std::ofstream file(filename);
    file << "Iteration,Time,Residual\n";
    for (size_t i = 0; i < iterations.size(); ++i) {
        file << iterations[i] << "," << times[i] << "," << residuals[i] << "\n";
    }
    file.close();
}



void compute_residuals_normalsolver_dense(const int dim, const int numIterations, std::mt19937& rng, std::vector<double>& times_residuals, std::vector<double>& residuals, std::vector<int>& iterations){
  for(int i = 0; i< numIterations; ++i){
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim); //Will need to be changed when known problems

    std::vector<double> x_kaczmarz(dim, 0.0);

        dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT * dim, PRECISION, times_residuals, residuals, iterations, CONVERGENCE_TRACKING_RATE);
  }
}

void compute_residuals_sparsesolver_sparse(const int dim, const int numIterations, std::mt19937& rng, std::vector<double>& times_residuals, std::vector<double>& residuals, std::vector<int>& iterations){
  for(int i = 0; i< numIterations; ++i){
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, BANDWIDTH);
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());

    sparse_kaczmarz(lse, x_kaczmarz_sparse, MAX_IT * dim, PRECISION, times_residuals, residuals, iterations, CONVERGENCE_TRACKING_RATE);
  }
}

void compute_residuals_randomsolver_dense(const int dim, const int numIterations, std::mt19937& rng, std::vector<double>& times_residuals, std::vector<double>& residuals, std::vector<int>& iterations){
  for(int i = 0; i< numIterations; ++i){
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim); //Will need to be changed when known problems

    std::vector<double> x_kaczmarz_random(dim, 0.0);

    kaczmarz_random_solver(lse, &x_kaczmarz_random[0], MAX_IT * dim, PRECISION,times_residuals, residuals, iterations, CONVERGENCE_TRACKING_RATE*200);
  }
}

int main() {
  std::mt19937 rng(21);

  std::vector<double> times_residuals_normal_dense;
  std::vector<double> residuals_normal_dense;
  std::vector<int> iterations_normal_dense;
  std::string filename_normal_dense="residuals_normalsolver_dense.csv";
  compute_residuals_normalsolver_dense(TEST_DIM, NUM_ITERATIONS, rng, times_residuals_normal_dense, residuals_normal_dense, iterations_normal_dense);

  export_convergence_to_csv(filename_normal_dense, iterations_normal_dense,times_residuals_normal_dense,residuals_normal_dense);


  std::vector<double> times_residuals_random_dense;
  std::vector<double> residuals_random_dense;
  std::vector<int> iterations_random_dense;
  std::string filename_random_dense="residuals_randomsolver_dense.csv";
  compute_residuals_randomsolver_dense(TEST_DIM, NUM_ITERATIONS, rng, times_residuals_random_dense, residuals_random_dense, iterations_random_dense);

  export_convergence_to_csv(filename_random_dense, iterations_random_dense,times_residuals_random_dense,residuals_random_dense);



  std::vector<double> times_residuals_sparse_sparse;
  std::vector<double> residuals_sparse_sparse;
  std::vector<int> iterations_sparse_sparse;
  std::string filename_sparse_sparse="residuals_sparsesolver_sparse.csv";
  compute_residuals_sparsesolver_sparse(TEST_DIM, NUM_ITERATIONS, rng, times_residuals_sparse_sparse, residuals_sparse_sparse, iterations_sparse_sparse);

  export_convergence_to_csv(filename_sparse_sparse, iterations_sparse_sparse,times_residuals_sparse_sparse,residuals_sparse_sparse);

  return 0;
}
