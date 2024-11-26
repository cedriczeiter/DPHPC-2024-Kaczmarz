#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "solvers/asynchronous.hpp"
#include "solvers/banded.hpp"
#include "solvers/basic.hpp"
#include "solvers/random.hpp"


#define MAX_IT 1000000
//#define BANDWIDTH 4
//#define MAX_DIM 512
#define PRECISION 1e-7
#define NUM_THREADS 8
//#define MIN_DIM 8
#define NUM_IT 4
//#define RANDOM_SEED 43

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

/// @brief Benchmarks the asynchronous Kaczmarz algorithm run on cuda on a
/// sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_asynchronouscuda_solver_sparse(const std::string& file_path,
                                                const int numIterations,
                                                double& stdDev) {
  std::vector<double> times;
    // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

  for (int i = 0; i < numIterations; ++i) {

    // Allocate memory to save kaczmarz solution
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    asynchronous_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                     NUM_THREADS);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the Kaczmarz algorithm run on cuda on a sparse linear
/// system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_asynchronouscpu_solver_sparse(const std::string& file_path,
                                               const int numIterations,
                                               double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {

    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    asynchronous_cpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                     NUM_THREADS);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the asynchronous Kaczmarz algorithm run on 2 cpu threads on a
/// sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_banded_2_cpu_threads_solver_sparse(const std::string& file_path,
                                                const int numIterations,
                                                double& stdDev) {
  std::vector<double> times;
    // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

      //still need to convert the SparseLinearSystem to BandedLinearSystem

  for (int i = 0; i < numIterations; ++i) {

    // Allocate memory to save kaczmarz solution
  Vector x_kaczmarz = Vector::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =  kaczmarz_banded_2_cpu_threads(lse, x_kaczmarz, MAX_IT,
    PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the asynchronous Kaczmarz algorithm run oncuda on a
/// sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_banded_cuda_solver_sparse(const std::string& file_path,
                                                const int numIterations,
                                                double& stdDev) {
  std::vector<double> times;
      // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
      //still need to convert the SparseLinearSystem to BandedLinearSystem

  for (int i = 0; i < numIterations; ++i) {

    // Allocate memory to save kaczmarz solution
  Vector x_kaczmarz = Vector::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =  kaczmarz_banded_cuda(lse, x_kaczmarz, MAX_IT,
    PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}


/// @brief Benchmarks the sparse Kaczmarz algorithm.
double benchmark_sparsesolver_sparse(const std::string& file_path, const int numIterations,
                                     double& stdDev) {
  std::vector<double> times;
      // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {

    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    sparse_kaczmarz(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                    times_residuals, residuals, iterations, 1000);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}


/// @brief Benchmarks Eigen solver on a sparse linear system.
double benchmark_EigenSolver_sparse(const std::string& file_path, const int numIterations,
                                    double& stdDev) {
  std::vector<double> times;
      // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {

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

/// @brief Benchmarks Eigen's iterative BiCGSTAB solver on a sparse linear system.
double benchmark_Eigeniterative_sparse(const std::string& file_path, const int numIterations,
                                    double& stdDev) {
  std::vector<double> times;
      // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {

    const auto start = std::chrono::high_resolution_clock::now();

    Vector x_kaczmarz_sparse = lse.eigen_BiCGSTAB();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

int main() {
  const int numIterations = NUM_IT;  // Number of iterations to reduce noise

//////////////////////////////////////////
  /// Cuda Banded Solver Sparse (Viktor)///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileBS1("results_banded_cuda_sparse_pde.csv");
  outFileBS1 << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime = benchmark_banded_cuda_solver_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileBS1 << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
    }
  }
  outFileBS1.close();  // Close the file after writing

//////////////////////////////////////////
  /// CPU 2 threads Banded Solver Sparse (Viktor)///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileBS2("results_banded_cpu_2_threads_sparse_pde.csv");
  outFileBS2 << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime = benchmark_banded_2_cpu_threads_solver_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileBS2 << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
  }
}
  outFileBS2.close();  // Close the file after writing

  ////////////////////////////////////////
  // Cuda Asynchronous Solver Sparse///
  ////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileNS1("results_asynchronous_cuda_sparse_pde.csv");
  outFileNS1 << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime = benchmark_asynchronouscuda_solver_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileNS1 << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
  }
}
  outFileNS1.close();  // Close the file after writing

  //////////////////////////////////////////
  /// CPU Asynchronous Solver Sparse///
  //////////////////////////////////////////

  // // Open the file for output
  // std::ofstream outFileNS2("results_asynchronous_cpu_sparse_pde.csv");
  // outFileNS2 << "Dim,AvgTime,StdDev\n";  // Write the header for the CSV file

  // // Loop over problem sizes, benchmark, and write to file
  // for (int complexity = 1; complexity <= 6; ++complexity) {
  //  std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
  //                        std::to_string(complexity) + ".txt";
  //   double stdDev;
  // try {
  //   double avgTime = benchmark_asynchronouscpu_solver_sparse(file_path, numIterations,
  //                                                            stdDev);

  //   // Write results to the file
  //   outFileNS2 << dim << "," << avgTime << "," << stdDev << "\n";
  // } catch (const std::exception& e) {
  //   std::cerr << "Error processing file " << file_path << ": " << e.what()
  //             << std::endl;
  // }
  // }
  // outFileNS2.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Normal Solver Sparse///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileNS3("results_sparsesolver_sparse_pde.csv");
  outFileNS3 << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime =
        benchmark_sparsesolver_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileNS3 << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
  }
}
  outFileNS3.close();  // Close the file after writing

  //////////////////////////////////////////
  /// Eigen Solver Sparse///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileES("results_eigensolver_sparse_pde.csv");
  outFileES << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime =
        benchmark_EigenSolver_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileES << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
  }
}
  outFileES.close();  // Close the file after writing

    //////////////////////////////////////////
  /// Eigen Iterative Solver Sparse///
  //////////////////////////////////////////

  // Open the file for output
  std::ofstream outFileEI("results_eigeniterative_sparse_pde.csv");
  outFileEI << "File,AvgTime,StdDev\n";  // Write the header for the CSV file

  // Loop over problem sizes, benchmark, and write to file
for (int complexity = 1; complexity <= 6; ++complexity) {
    std::string file_path = "../../generated_bvp_matrices/problem1_complexity" +
                          std::to_string(complexity) + ".txt";
    double stdDev;
     try {
    double avgTime =
        benchmark_Eigeniterative_sparse(file_path, 
        numIterations, stdDev);

    // Write results to the file
    outFileEI << file_path << "," << avgTime << "," << stdDev << "\n";
    } catch (const std::exception& e) {
    std::cerr << "Error processing file " << file_path << ": " << e.what()
              << std::endl;
  }
}
  outFileEI.close();  // Close the file after writing

  return 0;
}
