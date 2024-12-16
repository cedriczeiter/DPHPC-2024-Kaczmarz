#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
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
#include "solvers/carp.hpp"
#include "solvers/cusolver.hpp"
#include "solvers/random.hpp"
#include "solvers/sparse_cg.hpp"

#define MAX_IT (std::numeric_limits<unsigned int>::max()-1)
#define PRECISION 1e-7
#define NUM_IT 1
#define MAX_PROBLEMS 3
#define NR_OF_STEPS_CARP 0
#define RELAXATION 1

int compute_bandwidth(const Eigen::SparseMatrix<double>& A) {
  int bandwidth = 0;

  // Traverse each row (or column) of the sparse matrix
  for (int i = 0; i < A.outerSize(); ++i) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
      int row = it.row();  // Row index of the current nonzero entry
      int col = it.col();  // Column index of the current nonzero entry
      bandwidth = std::max(bandwidth, std::abs(row - col));
    }
  }

  return bandwidth;
}

BandedLinearSystem convert_to_banded(const SparseLinearSystem& sparse_system,
                                     unsigned bandwidth) {
  // Extract dimension
  unsigned dim = sparse_system.A().rows();

  // Ensure the sparse matrix is compressed
  Eigen::SparseMatrix<double> A_compressed = sparse_system.A();
  A_compressed.makeCompressed();

  // Prepare storage for banded matrix data
  std::vector<double> banded_data;
  banded_data.reserve(dim * (2 * bandwidth + 1) - bandwidth * (bandwidth + 1));

  // Fill the banded data using InnerIterator
  for (int k = 0; k < A_compressed.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(A_compressed, k); it;
         ++it) {
      int i = it.row();                         // Row index
      int j = it.col();                         // Column index
      if (std::abs(i - j) <= (int)bandwidth) {  // Check if within bandwidth
        banded_data.push_back(it.value());
      }
    }
  }

  // Initialize the BandedLinearSystem
  return BandedLinearSystem(dim, bandwidth, banded_data, sparse_system.b());
}

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

/// @brief Benchmarks the CARP Kaczmarz algorithm run on cuda on a
/// sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_carpcuda_solver_sparse(const std::string& file_path,
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
    int nr_of_steps = NR_OF_STEPS_CARP;  // just a placeholder, used in
                                         // benchmark_one_carp_lambda.cpp
    int relaxation = RELAXATION;         // just a placeholder, used in
                                         // benchmark_one_carp_lambda.cpp
    std::cout << "MAX IT "<< MAX_IT << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();

    // const auto status =
    //     carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION, relaxation,
    //     nr_of_steps);

    const auto status = carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION, relaxation,
             nr_of_steps);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
    if (status == KaczmarzSolverStatus::ZeroNormRow) {
      std::cout << "Zero norm row detected" << std::endl;
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      std::cout << "Max iterations reached" << std::endl;
    } else {
    }
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the asynchronous Kaczmarz algorithm run on 2 cpu threads
/// on a sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
/// @param rng Random generator for system generation.
/// @return Average time taken for solution.
double benchmark_banded_2_cpu_threads_solver_sparse(
    const std::string& file_path, const int numIterations, double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

  unsigned int bandwidth = compute_bandwidth(lse.A());
  BandedLinearSystem banded_lse = convert_to_banded(lse, bandwidth);
  // still need to convert the SparseLinearSystem to BandedLinearSystem

  for (int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Vector x_kaczmarz = Vector::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = kaczmarz_banded_2_cpu_threads(banded_lse, x_kaczmarz,
                                                      MAX_IT, PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the banded Kaczmarz algorithm run on cuda on a
/// sparse linear system.
/// @param dim Dimension of the system.
/// @param numIterations Number of iterations for timing.
/// @param stdDev Output parameter for the computed standard deviation.
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
  unsigned int bandwidth = compute_bandwidth(lse.A());
  BandedLinearSystem banded_lse = convert_to_banded(lse, bandwidth);
  // const BandedLinearSystem banded_lse(lse.row_count(),(unsigned int)
  // compute_bandwidth(lse.A()),
  //               lse.A(), lse.b());

  // still need to convert the SparseLinearSystem to BandedLinearSystem

  for (int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Vector x_kaczmarz = Vector::Zero(lse.column_count());

    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =
        kaczmarz_banded_cuda(banded_lse, x_kaczmarz, MAX_IT, PRECISION);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the sparse Kaczmarz algorithm.
double benchmark_sparsesolver_sparse(const std::string& file_path,
                                     const int numIterations, double& stdDev) {
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

    sparse_kaczmarz(lse, x_kaczmarz_sparse, MAX_IT, PRECISION, times_residuals,
                    residuals, iterations, 1000);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks the sparse cg Kaczmarz algorithm.
double benchmark_sparsesolver_cg(const std::string& file_path,
                                 const int numIterations, double& stdDev) {
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
    const auto start = std::chrono::high_resolution_clock::now();

    sparse_cg(lse, x_kaczmarz_sparse, PRECISION, MAX_IT);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks Eigen solver on a sparse linear system.
double benchmark_EigenSolver_sparse(const std::string& file_path,
                                    const int numIterations, double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  const auto A = lse.A();
  const auto b = lse.b();
  for (int i = 0; i < numIterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Vector x_kaczmarz_sparse = solver.solve(b);
    // Vector x_kaczmarz_sparse = lse.eigen_solve();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks Eigen's iterative BiCGSTAB solver on a sparse linear
/// system.
double benchmark_EigenBiCGSTAB_sparse(const std::string& file_path,
                                      const int numIterations, double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {
    const auto A = lse.A();
    const auto b = lse.b();
    Eigen::BiCGSTAB<SparseMatrix> solver(A);
    // lscg.preconditioner() = Eigen::IdentityPreconditioner;
    solver.setTolerance(PRECISION);
    solver.setMaxIterations(MAX_IT);
    const auto start = std::chrono::high_resolution_clock::now();

    Vector x_kaczmarz_sparse = solver.solve(b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks Eigen's iterative CG solver on a sparse linear system.
double benchmark_EigenCG_sparse(const std::string& file_path,
                                const int numIterations, double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  for (int i = 0; i < numIterations; ++i) {
    const auto A = lse.A();
    const auto b = lse.b();
    Eigen::LeastSquaresConjugateGradient<SparseMatrix> lscg(A);
    // lscg.preconditioner() = Eigen::IdentityPreconditioner;
    lscg.setTolerance(PRECISION);
    lscg.setMaxIterations(MAX_IT);
    const auto start = std::chrono::high_resolution_clock::now();

    Vector x_kaczmarz_sparse = lscg.solve(b);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

/// @brief Benchmarks cudas's solver on a sparse linear system.
double benchmark_cudadirect_sparse(const std::string& file_path,
                                   const int numIterations, double& stdDev) {
  std::vector<double> times;
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  const auto A = lse.A();
  const auto b = lse.b();
  for (int i = 0; i < numIterations; ++i) {
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    auto start = std::chrono::high_resolution_clock::now();

    KaczmarzSolverStatus status =
        cusolver(lse, x_kaczmarz_sparse, MAX_IT, PRECISION);

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  double avgTime = 0;
  compute_statistics(times, avgTime, stdDev);
  return avgTime;
}

void make_file_cuda_banded(const unsigned int min_problem,
                           const unsigned int max_problem,
                           const unsigned int min_complexity,
                           const unsigned int max_complexity,
                           const unsigned int min_degree,
                           const unsigned int max_degree,
                           const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_banded_cuda_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "BANDED GPU PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + "_banded.txt";
        double stdDev;
        try {
          double avgTime = benchmark_banded_cuda_solver_sparse(
              file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "BANDED SOLVER GPU IS DONE NOW" << std::endl;
}

void make_file_cpu_banded(const unsigned int min_problem,
                          const unsigned int max_problem,
                          const unsigned int min_complexity,
                          const unsigned int max_complexity,
                          const unsigned int min_degree,
                          const unsigned int max_degree,
                          const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_banded_cpu_2_threads_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "BANDED CPU PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + "_banded.txt";
        double stdDev;
        try {
          double avgTime = benchmark_banded_2_cpu_threads_solver_sparse(
              file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "BANDED SOLVER CPU IS DONE NOW" << std::endl;
}

void make_file_cuda_carp(const unsigned int min_problem,
                         const unsigned int max_problem,
                         const unsigned int min_complexity,
                         const unsigned int max_complexity,
                         const unsigned int min_degree,
                         const unsigned int max_degree,
                         const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_carp_cuda_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "CARP PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_carpcuda_solver_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing

  std::cout << "CARP IS DONE NOW" << std::endl;
}

void make_file_normal_solver(const unsigned int min_problem,
                             const unsigned int max_problem,
                             const unsigned int min_complexity,
                             const unsigned int max_complexity,
                             const unsigned int min_degree,
                             const unsigned int max_degree,
                             const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_sparsesolver_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "NORMAL SEQUENTIAL SOLVER PROBLEM " << problem_i
                  << " COMPLEXITY " << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_sparsesolver_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "NORMAL SEQUENTIAL SOLVER IS DONE NOW" << std::endl;
}

void make_file_sparse_cg(const unsigned int min_problem,
                         const unsigned int max_problem,
                         const unsigned int min_complexity,
                         const unsigned int max_complexity,
                         const unsigned int min_degree,
                         const unsigned int max_degree,
                         const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_sparsesolver_sparse_cg_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "SEQUENTIAL SOLVER CG PROBLEM " << problem_i
                  << " COMPLEXITY " << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_sparsesolver_cg(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "NORMAL SEQUENTIAL SOLVER IS DONE NOW" << std::endl;
}

void make_file_eigen_solver(const unsigned int min_problem,
                            const unsigned int max_problem,
                            const unsigned int min_complexity,
                            const unsigned int max_complexity,
                            const unsigned int min_degree,
                            const unsigned int max_degree,
                            const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_eigensolver_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file
  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "EIGEN SOLVER PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_EigenSolver_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "EIGEN NON ITERATIVE IS DONE NOW" << std::endl;
}

void make_file_eigen_iterative(const unsigned int min_problem,
                               const unsigned int max_problem,
                               const unsigned int min_complexity,
                               const unsigned int max_complexity,
                               const unsigned int min_degree,
                               const unsigned int max_degree,
                               const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_eigeniterative_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "EIGEN ITERATIVE PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_EigenCG_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "EIGEN ITERATIVE IS DONE NOW" << std::endl;
}

void make_file_eigen_iterative_better(const unsigned int min_problem,
                                      const unsigned int max_problem,
                                      const unsigned int min_complexity,
                                      const unsigned int max_complexity,
                                      const unsigned int min_degree,
                                      const unsigned int max_degree,
                                      const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_eigeniterative_2_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "EIGEN ITERATIVE BiCGSTAB PROBLEM " << problem_i
                  << " COMPLEXITY " << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_EigenBiCGSTAB_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "EIGEN ITERATIVE BiCGSTAB IS DONE NOW" << std::endl;
}

void make_file_cuda_direct(const unsigned int min_problem,
                           const unsigned int max_problem,
                           const unsigned int min_complexity,
                           const unsigned int max_complexity,
                           const unsigned int min_degree,
                           const unsigned int max_degree,
                           const unsigned int iterations) {
  // Open the file for output
  std::ofstream outFile("results_cudadirect_sparse_pde.csv");
  outFile
      << "File,Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                 // header for
                                                                 // the CSV file

  for (unsigned int problem_i = min_problem; problem_i <= max_problem;
       ++problem_i) {
    // Loop over problem sizes, benchmark, and write to file
    for (unsigned int complexity = min_complexity; complexity <= max_complexity;
         ++complexity) {
      for (unsigned int degree = min_degree; degree <= max_degree; ++degree) {
        std::cout << "CUDA DIRECT PROBLEM " << problem_i << " COMPLEXITY "
                  << complexity << " DEGREE " << degree
                  << " is being worked on now!" << std::endl;
        std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(problem_i) + "/problem" +
                                std::to_string(problem_i) + "_complexity" +
                                std::to_string(complexity) + "_degree" +
                                std::to_string(degree) + ".txt";
        double stdDev;
        try {
          double avgTime =
              benchmark_cudadirect_sparse(file_path, iterations, stdDev);

          unsigned nnz, rows, cols;
          std::ifstream lse_input_stream(file_path);
          if (!lse_input_stream) {
            throw std::runtime_error("Failed to open matrix file: " +
                                     file_path);
          }
          lse_input_stream >> nnz >> rows >> cols;
          // Write results to the file
          outFile << file_path << "," << problem_i << "," << complexity << ","
                  << degree << "," << avgTime << "," << stdDev << "," << rows
                  << "\n";
        } catch (const std::exception& e) {
          std::cerr << "Error processing file " << file_path << ": " << e.what()
                    << std::endl;
        }
      }
    }
  }
  outFile.close();  // Close the file after writing
  std::cout << "CUDA DIRECT IS DONE NOW" << std::endl;
}

int main() {
  unsigned int numIterations= 1;
          std::string file_path = "../../generated_bvp_matrices/problem" +
                                std::to_string(1) + "/problem" +
                                std::to_string(1) + "_complexity" +
                                std::to_string(1) + "_degree" +
                                std::to_string(1) + ".txt";
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

    int nr_of_steps = 0;  // just a placeholder, used in
                                         // benchmark_one_carp_lambda.cpp
    int relaxation = 0.35;         // just a placeholder, used in
                                         // benchmark_one_carp_lambda.cpp
    std::cout << "MAX IT "<< MAX_IT << std::endl;
    const auto start = std::chrono::high_resolution_clock::now();

    // const auto status =
    //     carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION, relaxation,
    //     nr_of_steps);

    const auto status = carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION, relaxation,
             nr_of_steps);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
    if (status == KaczmarzSolverStatus::ZeroNormRow) {
      std::cout << "Zero norm row detected" << std::endl;
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      std::cout << "Max iterations reached" << std::endl;
    } else {
    }
  }

  //make_file_cuda_carp(1, MAX_PROBLEMS, 1, 2, 1, 1, NUM_IT);
  // make_file_eigen_solver(1, MAX_PROBLEMS, 1, 6, 1, 1, NUM_IT);
  // make_file_cuda_direct(1, MAX_PROBLEMS, 1, 5, 1, 1, NUM_IT);
  // make_file_eigen_iterative(1, MAX_PROBLEMS, 1, 6, 1, 1, NUM_IT);
  // make_file_eigen_iterative_better(1, MAX_PROBLEMS, 1, 6, 1, 1, NUM_IT);
  // make_file_normal_solver(1, MAX_PROBLEMS, 1, 4, 1, 1, NUM_IT);
  // make_file_sparse_cg(1, MAX_PROBLEMS, 1, 6, 1, 1, NUM_IT);
  // make_file_cuda_banded(1, MAX_PROBLEMS, 1, 3, 1, 1, NUM_IT);
  // make_file_cpu_banded(1, MAX_PROBLEMS, 1, 3, 1, 1, NUM_IT);

  return 0;
}
