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
#define MAX_DIM 16
#define PRECISION 1e-10

/// @brief Computes the average and standard deviation of a vector of times.
/// @param times A vector of times recorded for benchmarking in seconds.
/// @param avgTime Reference to store the computed average time in seconds.
/// @param stdDev Reference to store the computed standard deviation.
void compute_statistics(const std::vector<double>& times, double& avgTime,
                        double& stdDev) {
  int n = times.size();
  if (n == 0) {
    avgTime = 0;
    stdDev = 0;
    return;
  }
  avgTime = std::accumulate(times.begin(), times.end(), 0.0) / n;

  double variance = 0;
  for (double time : times) {
    variance += (time - avgTime) * (time - avgTime);
  }
  variance /= n;
  stdDev = std::sqrt(variance);
}

/// @brief Benchmarks the Kaczmarz algorithm on a dense linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_normalsolver_dense(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim);

    // Allocate memory to save kaczmarz solution
    std::vector<double> x_kaczmarz(dim, 0.0);

    const auto start = std::chrono::high_resolution_clock::now();

    dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT * dim, PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the sparse Kaczmarz algorithm.
double benchmark_sarsesolver_sparse(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, BANDWIDTH);

    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());

    const auto start = std::chrono::high_resolution_clock::now();

    sparse_kaczmarz(lse, x_kaczmarz_sparse, MAX_IT * dim, PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the random Kaczmarz solver on a dense linear system.
double benchmark_randomsolver_dense(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim);

    std::vector<double> x_kaczmarz_random(dim, 0.0);

    const auto start = std::chrono::high_resolution_clock::now();

    kaczmarz_random_solver(lse, &x_kaczmarz_random[0], MAX_IT * dim, PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks Eigen solver on a sparse linear system.
double benchmark_EigenSolver_sparse(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, BANDWIDTH);

    const auto start = std::chrono::high_resolution_clock::now();

    Vector x_kaczmarz_sparse = lse.eigen_solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks Eigen solver on a dense linear system.
double benchmark_EigenSolver_dense(const int dim, const int numIterations,
                                   double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim);

    const auto start = std::chrono::high_resolution_clock::now();

    Vector x_kaczmarz = lse.eigen_solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

int main() {
  const int numIterations = 10;  // Number of iterations to reduce noise
  std::mt19937 rng(21);

  //////////////////////////////////////////
  /// Normal Solver Dense///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileND("results_normalsolver_dense.csv");
  outFileND << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_normalsolver_dense(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileND << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileND.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Normal Solver Sparse///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileNS("results_sparsesolver_sparse.csv");
  outFileNS << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_sarsesolver_sparse(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileNS << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileNS.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Random Solver Dense///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileRD("results_randomsolver_dense.csv");
  outFileRD << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_randomsolver_dense(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileRD << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileRD.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Eigen Solver Dense///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileED("results_eigensolver_dense.csv");
  outFileED << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_EigenSolver_dense(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileED << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileED.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Eigen Solver Sparse///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileES("results_eigensolver_sparse.csv");
  outFileES << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_EigenSolver_sparse(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileES << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileES.close();  // Close the file after writing

  return 0;
}
