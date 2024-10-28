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

#define MAX_IT 1000000
#define BANDWIDTH 5
#define MAX_DIM 16

/// @brief Benchmarks the Kaczmarz algorithm on a randomly generated dense
/// linear system.
///
/// This function generates a random dense linear system of a given dimension
/// and solves it using the Kaczmarz algorithm multiple times to gather
/// statistics on average runtime and standard deviation.
///
/// @param dim The dimension of the linear system (number of variables).
/// @param numIterations Number of iterations for averaging results.
/// @param stdDev Reference to a variable where the computed standard deviation
/// will be stored.
/// @param rng A random number generator used for creating the linear system.
/// @return The average time taken to solve the system across all iterations.
double benchmark_normalsolver_dense(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim);

    // Allocate memory to save kaczmarz solution
    std::vector<double> x_kaczmarz(dim, 0.0);

    const auto start = std::chrono::high_resolution_clock::now();

    dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT * dim,
                   1e-10);  // solve randomised system, max iterations steps
                            // selected arbitratly, we might need to revise this

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  // Calculate average time
  double avgTime = 0;
  for (double time : times) {
    avgTime += time;
  }
  avgTime /= numIterations;

  // Calculate standard deviation
  double variance = 0;
  for (double time : times) {
    variance += (time - avgTime) * (time - avgTime);
  }
  variance /= numIterations;
  stdDev = std::sqrt(variance);

  return avgTime;
}

double benchmark_sarsesolver_sparse(const int dim, const int numIterations,
                                    double& stdDev, std::mt19937& rng) {
  std::vector<double> times;
  for (int i = 0; i < numIterations; ++i) {
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, BANDWIDTH);

    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());

    const auto start = std::chrono::high_resolution_clock::now();

    sparse_kaczmarz(
        lse, x_kaczmarz_sparse, MAX_IT * dim,
        1e-10);  // solve randomised system, max iterations steps
                 // selected arbitratly, we might need to revise this

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  // Calculate average time
  double avgTime = 0;
  for (double time : times) {
    avgTime += time;
  }
  avgTime /= numIterations;

  // Calculate standard deviation
  double variance = 0;
  for (double time : times) {
    variance += (time - avgTime) * (time - avgTime);
  }
  variance /= numIterations;
  stdDev = std::sqrt(variance);

  return avgTime;
}

int main() {
  const int numIterations = 10;  // Number of iterations to reduce noise
  std::mt19937 rng(21);

  //////////////////////////////////////////
  /// Normal Sovler Dense///
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
  /// Normal Sovler Dense///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileSS("results_sparsesolver_sparse.csv");
  outFileSS << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
  for (int dim = 1; dim <= MAX_DIM; dim *= 2) {
    double stdDev;
    double avgTime =
        benchmark_sarsesolver_sparse(dim, numIterations, stdDev, rng);

    // Write results to the file
    outFileSS << dim << "," << avgTime << "," << stdDev << "\n";
  }
  outFileSS.close();  // Close the file after writing
  return 0;
}
