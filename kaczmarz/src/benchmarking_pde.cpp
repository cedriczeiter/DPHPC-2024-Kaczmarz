#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
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

#define MAX_IT (std::numeric_limits<unsigned int>::max() - 1)
#define PRECISION 1e-9
#define MAX_PROBLEMS 3
#define MAX_COMPLEXITY 8
#define MAX_DEGREE 1
#define NUM_IT 10

// Function declarations
void write_header(const std::string& file_path);
int compute_bandwidth(const Eigen::SparseMatrix<double>& A);
BandedLinearSystem convert_to_banded(const SparseLinearSystem& sparse_system,
                                     unsigned bandwidth);
void compute_statistics(const std::vector<double>& times, double& avgTime,
                        double& stdDev);
void write_results_to_file(const std::string& file_name, unsigned int problem,
                           unsigned int complexity, unsigned int degree,
                           double avg_time, double std_dev,
                           unsigned int dimension);

double benchmark_carpcg(unsigned int numIterations, unsigned int problem_i,
                        unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_cg(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_bicgstab(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i);
double benchmark_cgmnc(unsigned int numIterations, unsigned int problem_i,
                       unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_direct(unsigned int numIterations,
                              unsigned int problem_i, unsigned int complexity_i,
                              unsigned int degree_i);
double benchmark_basic_kaczmarz(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i);
double benchmark_cusolver(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_cuda(unsigned int numIterations, unsigned int problem_i,
                             unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_cpu(unsigned int numIterations, unsigned int problem_i,
                            unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_serial(unsigned int numIterations,
                               unsigned int problem_i,
                               unsigned int complexity_i,
                               unsigned int degree_i);

int main() {
  // Define a threshold in seconds
  const double TIME_THRESHOLD = 500.0;
  // Map to track execution times of algorithms

  std::unordered_map<std::string,
                     std::function<double(unsigned int, unsigned int,
                                          unsigned int, unsigned int)>>
      algorithms;
  std::vector<std::string> algorithms_names = {
      "CARP_CG",      "Eigen_CG",       "Eigen_BiCGSTAB", "CGMNC",
      "Eigen_Direct", "Basic_Kaczmarz", /*"Banded_CPU", "Banded_CUDA",
                                           "Banded_SERIAL", */
      "CUSolver"};
  algorithms = {{"CARP_CG", benchmark_carpcg},
                {"Eigen_CG", benchmark_eigen_cg},
                {"Eigen_BiCGSTAB", benchmark_eigen_bicgstab},
                {"CGMNC", benchmark_cgmnc},
                {"Eigen_Direct", benchmark_eigen_direct},
                {"Basic_Kaczmarz", benchmark_basic_kaczmarz},
                {"Banded_CPU", benchmark_banded_cpu},
                {"Banded_CUDA", benchmark_banded_cuda},
                {"Banded_SERIAL", benchmark_banded_serial},
                {"CUSolver", benchmark_cusolver}};
  std::vector<std::string> file_names = {
      "results_banded_serial_sparse_pde.csv",
      "results_banded_cpu_2_threads_sparse_pde.csv",
      "results_banded_cuda_sparse_pde.csv",
      "results_cudadirect_sparse_pde.csv",
      "results_sparsesolver_sparse_pde.csv",
      "results_eigensolver_sparse_pde.csv",
      "results_sparsesolver_sparse_cg_pde.csv",
      "results_eigeniterative_2_sparse_pde.csv",
      "results_eigeniterative_sparse_pde.csv",
      "results_carp_cuda_sparse_pde.csv"};

  // Loop over file names and call write_header
  for (const auto& file_name : file_names) {
    write_header(file_name);
  }
  unsigned int iterations = NUM_IT;
  // Main loop over degrees
  for (unsigned int degree = 1; degree <= MAX_DEGREE; ++degree) {
    std::cout << "Processing degree: " << degree << std::endl;
    std::unordered_map<
        std::string,
        std::unordered_map<unsigned int,
                           std::unordered_map<unsigned int, double>>>
        execution_times;

    // Loop over complexities
    for (unsigned int complexity = 1; complexity <= MAX_COMPLEXITY;
         ++complexity) {
      std::cout << "  Processing complexity: " << complexity << std::endl;

      // Create a list of problems
      std::vector<unsigned int> problems = {1, 2, 3};

      // Randomize algorithm order for this complexity
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(algorithms_names.begin(), algorithms_names.end(), g);

      // Execute each problem for the current complexity and degree
      for (unsigned int problem : problems) {
        std::cout << "    Processing problem: " << problem << std::endl;

        for (auto& algorithm_name : algorithms_names) {
          // Skip higher complexities if the algorithm exceeded the threshold
          if (execution_times.count(algorithm_name) > 0 &&
              execution_times[algorithm_name].count(problem) > 0 &&
              execution_times[algorithm_name][problem].count(complexity - 1) >
                  0 &&
              execution_times[algorithm_name][problem][complexity - 1] >
                  TIME_THRESHOLD) {
            std::cout << "      Skipping " << algorithm_name
                      << " for complexity " << complexity
                      << " due to high execution time at a lower complexity."
                      << std::endl;
            continue;
          }
          try {
            double time =
                algorithms[algorithm_name](iterations, problem, complexity,
                                           degree);  // Run the algorithm
                                                     // Record execution time
            execution_times[algorithm_name][problem][complexity] = time;
          } catch (const std::exception& e) {
            std::cerr << "Error in algorithm execution for problem " << problem
                      << ", complexity " << complexity << ", degree " << degree
                      << ": " << e.what() << std::endl;
          }
        }
      }
    }
  }

  return 0;
}

void write_header(const std::string& file_path) {
  // Open the file in truncation mode to clear existing content
  std::ofstream outFile(
      file_path);  // Default mode is std::ios::out | std::ios::trunc
  if (outFile.is_open()) {
    outFile << "Problem,Complexity,Degree,AvgTime,StdDev,Dim\n";  // Write the
                                                                  // header
    outFile.flush();  // Ensure data is written to the disk immediately
    outFile.close();
  } else {
    std::cerr << "Error: Unable to open file " << file_path << " for writing."
              << std::endl;
  }
}

// Utility function to write results to files
void write_results_to_file(const std::string& file_name, unsigned int problem,
                           unsigned int complexity, unsigned int degree,
                           double avg_time, double std_dev,
                           unsigned int dimension) {
  std::ofstream outFile(file_name, std::ios::app);
  outFile << problem << "," << complexity << "," << degree << "," << avg_time
          << "," << std_dev << "," << dimension << "\n";
  outFile.flush();  // Ensure data is written to the disk immediately
  outFile.close();
}

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

double benchmark_carpcg(unsigned int numIterations, unsigned int problem_i,
                        unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running CARP for problem " << problem_i << ", complexity "
            << complexity_i << ", degree " << degree_i << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    int nr_of_steps = 0;       // just a placeholder, used in
                               // benchmark_one_carp_lambda.cpp
    double relaxation = 0.35;  // just a placeholder, used in
                               // benchmark_one_carp_lambda.cpp
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                                 relaxation, nr_of_steps);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_carp_cuda_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_eigen_cg(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running EIGEN CG for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_eigeniterative_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_eigen_bicgstab(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running EIGEN BiCGSTAB for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_eigeniterative_2_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_cgmnc(unsigned int numIterations, unsigned int problem_i,
                       unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running CGMNC for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = sparse_cg(lse, x_kaczmarz_sparse, PRECISION, MAX_IT);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_sparsesolver_sparse_cg_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_eigen_direct(unsigned int numIterations,
                              unsigned int problem_i, unsigned int complexity_i,
                              unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();
  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running EIGEN DIRECT for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  const auto A = lse.A();
  const auto b = lse.b();
  for (unsigned int i = 0; i < numIterations; ++i) {
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    const auto start = std::chrono::high_resolution_clock::now();
    Vector x_kaczmarz_sparse = solver.solve(b);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    times.push_back(elapsed.count());
  }

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_eigensolver_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}
double benchmark_basic_kaczmarz(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();
  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running BASIC KACZMARZ for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =
        sparse_kaczmarz(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                        times_residuals, residuals, iterations, 1000);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_sparsesolver_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_cusolver(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + ".txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running CUSOLVER for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  const auto A = lse.A();
  const auto b = lse.b();
  for (unsigned int i = 0; i < numIterations; ++i) {
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    auto start = std::chrono::high_resolution_clock::now();

    KaczmarzSolverStatus status =
        cusolver(lse, x_kaczmarz_sparse, MAX_IT, PRECISION);

    // End timer
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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_cudadirect_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_banded_cuda(unsigned int numIterations, unsigned int problem_i,
                             unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + "_banded.txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running BANDED CUDA for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  unsigned int bandwidth = compute_bandwidth(lse.A());
  BandedLinearSystem banded_lse = convert_to_banded(lse, bandwidth);

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Vector x_kaczmarz = Vector::Zero(lse.column_count());

    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =
        kaczmarz_banded_cuda(banded_lse, x_kaczmarz, MAX_IT, PRECISION);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_banded_cuda_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}

double benchmark_banded_cpu(unsigned int numIterations, unsigned int problem_i,
                            unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + "_banded.txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running BANDED CPU for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  unsigned int bandwidth = compute_bandwidth(lse.A());
  BandedLinearSystem banded_lse = convert_to_banded(lse, bandwidth);

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Vector x_kaczmarz = Vector::Zero(lse.column_count());
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = kaczmarz_banded_2_cpu_threads(banded_lse, x_kaczmarz,
                                                      MAX_IT, PRECISION);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_banded_cpu_2_threads_sparse_pde.csv",
                        problem_i, complexity_i, degree_i, avg_time, std_dev,
                        rows);
  return avg_time;
}
double benchmark_banded_serial(unsigned int numIterations,
                               unsigned int problem_i,
                               unsigned int complexity_i,
                               unsigned int degree_i) {
  std::string file_path = "../../generated_bvp_matrices/problem" +
                          std::to_string(problem_i) + "/problem" +
                          std::to_string(problem_i) + "_complexity" +
                          std::to_string(complexity_i) + "_degree" +
                          std::to_string(degree_i) + "_banded.txt";
  // Read the precomputed matrix from the file
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();

  double avg_time = 0.0, std_dev = 0.0;
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running BANDED SERIAL for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  unsigned int bandwidth = compute_bandwidth(lse.A());
  BandedLinearSystem banded_lse = convert_to_banded(lse, bandwidth);

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Vector x_kaczmarz = Vector::Zero(lse.column_count());
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status =
        kaczmarz_banded_serial(banded_lse, x_kaczmarz, MAX_IT, PRECISION);

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

  compute_statistics(times, avg_time, std_dev);
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream_2(file_path);
  if (!lse_input_stream_2) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream_2 >> nnz >> rows >> cols;
  lse_input_stream_2.close();

  // Write results
  write_results_to_file("results_banded_serial_sparse_pde.csv", problem_i,
                        complexity_i, degree_i, avg_time, std_dev, rows);
  return avg_time;
}