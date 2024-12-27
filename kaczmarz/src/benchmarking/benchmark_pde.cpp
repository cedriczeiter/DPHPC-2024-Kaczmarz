#include "benchmark_pde.hpp"

int main() {
  std::unordered_map<std::string,
                     std::function<double(unsigned int, unsigned int,
                                          unsigned int, unsigned int)>>
      algorithms;

  algorithms = {{"CARP_CG", benchmark_carpcg},
                {"Eigen_CG", benchmark_eigen_cg},
                {"Eigen_BiCGSTAB", benchmark_eigen_bicgstab},
                {"CGMNC", benchmark_cgmnc},
                {"Eigen_Direct", benchmark_eigen_direct},
                {"Basic_Kaczmarz", benchmark_basic_kaczmarz},
                {"CUSolver", benchmark_cusolver}};

  std::vector<std::string> file_names = {
      "results_cudadirect_sparse_pde.csv",
      "results_sparsesolver_sparse_pde.csv",
      "results_eigensolver_sparse_pde.csv",
      "results_sparsesolver_sparse_cg_pde.csv",
      "results_eigeniterative_2_sparse_pde.csv",
      "results_eigeniterative_sparse_pde.csv",
      "results_carp_cuda_sparse_pde.csv"};

  // Loop over file names and call write_header
  for (const auto &file_name : file_names) {
    write_header(file_name);
  }
  // Main loop over degrees
  for (unsigned int degree = 1; degree <= MAX_DEGREE; ++degree) {
    std::cout << "Processing degree: " << degree << std::endl;
    std::unordered_map<std::string, std::unordered_map<unsigned int, bool>>
        skip_algorithm;

    // Loop over complexities
    for (unsigned int complexity = 1; complexity <= MAX_COMPLEXITY;
         ++complexity) {
      std::cout << "  Processing complexity: " << complexity << std::endl;

      // Create a list of problems
      std::vector<unsigned int> problems = {1, 2, 3};
      std::vector<std::string> algorithms_names;

      if (complexity <= 4) {
        // These algorithms are used for benchmarking
        algorithms_names = {"CARP_CG", "Eigen_CG",     "Eigen_BiCGSTAB",
                            "CGMNC",   "Eigen_Direct", "Basic_Kaczmarz",
                            "CUSolver"};
      } else {
        // These algorithms are used for benchmarking, we don't benchmark Basic
        // Kaczmarz anymore, if higher complexity than 4
        algorithms_names = {"CARP_CG", "Eigen_CG",     "Eigen_BiCGSTAB",
                            "CGMNC",   "Eigen_Direct", "CUSolver"};
      }

      // Randomize algorithm order for this complexity
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(algorithms_names.begin(), algorithms_names.end(), g);
      unsigned int num_it = 0;

      // Slowly decrease the number of iterations. Goal is to not use more than
      // 10 minutes for each of the lower complexities. To solve each of the
      // problems for a complexity this should take accross all methods around
      // 60 minutes (1 hour). So three problems will take around 3 hours per
      // complexity. This will hold up to around complexity 6, so 18 hours to
      // get to the end of complexity 6. Then complexity 7 will take around 33
      // minutes per method, so around 180 minutes per problem, so a total of
      // around 9 hours. Complexity 8 will take around 166 minutes per method,
      // so around 1000 minutes per problem = 16 hours. So around 48 hours to
      // get to the end of complexity 8. (very pessimistic estimate)
      switch (complexity) {
        // For the first four complexities, we use 200 iterations, because the
        // worst algorithm takes around 0.1 seconds. 6000*0.1 = 600 seconds = 10
        // minutes
        case 1:
        case 2:
        case 3:
        case 4:
          num_it = 6000;
          break;
        // For the next complexities, we use 60 iterations, because the worst
        // algorithm takes around 10 seconds. 60*10 = 600 seconds = 10 minutes
        case 5:
        case 6:
          num_it = 60;
          break;
          // For the the next complexity, we use 20 iterations, because the
          // worst algorithm takes around 100 seconds 20*100 = 2000 seconds = 33
          // minutes
        case 7:
          num_it = 20;
          break;
          // For the last complexity, we use 10 iterations, because the worst
          // algorithm takes around 1000 seconds 10*1000 = 10000 seconds = 166
          // minutes
        case 8:
          num_it = 10;
          break;
        default:
          num_it = NUM_IT;
      }

      // Execute each problem for the current complexity and degree
      for (unsigned int problem : problems) {
        std::cout << "    Processing problem: " << problem << std::endl;

        for (auto &algorithm_name : algorithms_names) {
          // Check if the algorithm is already marked to skip (if true stored)
          if (skip_algorithm[algorithm_name][problem]) {
            std::cout << "      Skipping " << algorithm_name
                      << " for complexity " << complexity
                      << " due to previous high execution time." << std::endl;
            continue;  // Skip this algorithm for all higher complexities
          }
          try {
            double time =
                algorithms[algorithm_name](num_it, problem, complexity,
                                           degree);  // Run the algorithm
                                                     // Record execution time
            // Check if the execution time exceeds the threshold
            if (time > TIME_THRESHOLD) {
              std::cout << "      Marking " << algorithm_name
                        << " for skipping due to execution time: " << time
                        << " seconds." << std::endl;
              skip_algorithm[algorithm_name][problem] = true;
            }
          } catch (const std::exception &e) {
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

///////////////////////////////////////////
// Benchmarking functions
///////////////////////////////////////////

double benchmark_carpcg(unsigned int numIterations, unsigned int problem_i,
                        unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  std::cout << "      Running CARP Cuda Sparse for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    int nr_of_steps = 0;  // just a placeholder, gets overwritten with nr of
                          // steps, but not used here
    double relaxation = 0.35;  // found to be optimal by experiments
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = carp_gpu(lse, x_kaczmarz_sparse, MAX_IT, PRECISION,
                                 relaxation, nr_of_steps);

    const auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    if (status == KaczmarzSolverStatus::Converged) {
      write_result_to_file("results_carp_cuda_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Converged");
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      write_result_to_file("results_carp_cuda_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "OutOfIterations");
    } else if (status == KaczmarzSolverStatus::ZeroNormRow) {
      write_result_to_file("results_carp_cuda_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "ZeroNormRow");
    } else {
      write_result_to_file("results_carp_cuda_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Failed");
    }
    add_elapsed_time_to_vec(times, start, end);

    // inform_user_about_kaczmarz_status(status);
  }
  return calc_avgtime(times);
}

double benchmark_eigen_cg(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;
  const auto A = lse.A();
  const auto b = lse.b();
  // Perform benchmarking
  std::cout << "      Running EIGEN Iterative sparse CG for problem "
            << problem_i << ", complexity " << complexity_i << ", degree "
            << degree_i << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    const auto start = std::chrono::high_resolution_clock::now();
    Eigen::LeastSquaresConjugateGradient<SparseMatrix> lscg(A);
    // lscg.preconditioner() = Eigen::IdentityPreconditioner;
    lscg.setTolerance(PRECISION);
    lscg.setMaxIterations(MAX_IT);

    Vector x_kaczmarz_sparse = lscg.solve(b);

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    write_result_to_file("results_eigeniterative_sparse_pde.csv", problem_i,
                         complexity_i, degree_i, elapsed.count(), dimension,
                         numIterations, i, "Converged");
    add_elapsed_time_to_vec(times, start, end);
  }

  return calc_avgtime(times);
}

double benchmark_eigen_bicgstab(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  const auto A = lse.A();
  const auto b = lse.b();

  // Perform benchmarking
  std::cout << "      Running EIGEN iterative 2 BiCGSTAB for problem "
            << problem_i << ", complexity " << complexity_i << ", degree "
            << degree_i << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    Eigen::BiCGSTAB<SparseMatrix> solver(A);
    // lscg.preconditioner() = Eigen::IdentityPreconditioner;
    solver.setTolerance(PRECISION);
    solver.setMaxIterations(MAX_IT);

    Vector x_kaczmarz_sparse = solver.solve(b);

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    write_result_to_file("results_eigeniterative_2_sparse_pde.csv", problem_i,
                         complexity_i, degree_i, elapsed.count(), dimension,
                         numIterations, i, "Converged");
    add_elapsed_time_to_vec(times, start, end);
  }

  return calc_avgtime(times);
}

double benchmark_cgmnc(unsigned int numIterations, unsigned int problem_i,
                       unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running CGMNC (Sparse CG) for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  for (unsigned int i = 0; i < numIterations; ++i) {
    // Allocate memory to save kaczmarz solution
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    const auto start = std::chrono::high_resolution_clock::now();

    const auto status = sparse_cg(lse, x_kaczmarz_sparse, PRECISION, MAX_IT);

    const auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    if (status == KaczmarzSolverStatus::Converged) {
      write_result_to_file("results_sparsesolver_sparse_cg_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Converged");
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      write_result_to_file("results_sparsesolver_sparse_cg_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "OutOfIterations");
    } else if (status == KaczmarzSolverStatus::ZeroNormRow) {
      write_result_to_file("results_sparsesolver_sparse_cg_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "ZeroNormRow");

    } else {
      write_result_to_file("results_sparsesolver_sparse_cg_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Failed");
    }
    add_elapsed_time_to_vec(times, start, end);

    inform_user_about_kaczmarz_status(status);
  }

  return calc_avgtime(times);
}

double benchmark_eigen_direct(unsigned int numIterations,
                              unsigned int problem_i, unsigned int complexity_i,
                              unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running EIGEN DIRECT sparse for problem " << problem_i
            << ", complexity " << complexity_i << ", degree " << degree_i
            << std::endl;

  const auto A = lse.A();
  const auto b = lse.b();
  for (unsigned int i = 0; i < numIterations; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    Vector x_kaczmarz_sparse = solver.solve(b);

    const auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    write_result_to_file("results_eigensolver_sparse_pde.csv", problem_i,
                         complexity_i, degree_i, elapsed.count(), dimension,
                         numIterations, i, "Converged");

    add_elapsed_time_to_vec(times, start, end);
  }

  return calc_avgtime(times);
}
double benchmark_basic_kaczmarz(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running BASIC KACZMARZ (sparsesolver sparse) for problem "
            << problem_i << ", complexity " << complexity_i << ", degree "
            << degree_i << std::endl;

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

    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (status == KaczmarzSolverStatus::Converged) {
      write_result_to_file("results_sparsesolver_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Converged");
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      write_result_to_file("results_sparsesolver_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "OutOfIterations");
    } else if (status == KaczmarzSolverStatus::ZeroNormRow) {
      write_result_to_file("results_sparsesolver_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "ZeroNormRow");
    } else {
      write_result_to_file("results_sparsesolver_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Failed");
    }
    add_elapsed_time_to_vec(times, start, end);

    inform_user_about_kaczmarz_status(status);
  }

  return calc_avgtime(times);
}

double benchmark_cusolver(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i) {
  std::string file_path = generate_file_path(problem_i, complexity_i, degree_i);
  // Read the precomputed matrix from the file
  const SparseLinearSystem lse = read_matrix_from_file(file_path);
  const unsigned int dimension = read_dimension(file_path);
  std::vector<double> times;

  // Perform benchmarking
  std::cout << "      Running CUSOLVER (cudadirect sparse) for problem "
            << problem_i << ", complexity " << complexity_i << ", degree "
            << degree_i << std::endl;

  const auto A = lse.A();
  const auto b = lse.b();
  for (unsigned int i = 0; i < numIterations; ++i) {
    Eigen::VectorXd x_kaczmarz_sparse =
        Eigen::VectorXd::Zero(lse.column_count());
    auto start = std::chrono::high_resolution_clock::now();

    KaczmarzSolverStatus status =
        cusolver(lse, x_kaczmarz_sparse, MAX_IT, PRECISION);

    // End timer
    const auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    if (status == KaczmarzSolverStatus::Converged) {
      write_result_to_file("results_cudadirect_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Converged");
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      write_result_to_file("results_cudadirect_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "OutOfIterations");
    } else if (status == KaczmarzSolverStatus::ZeroNormRow) {
      write_result_to_file("results_cudadirect_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "ZeroNormRow");
    } else {
      write_result_to_file("results_cudadirect_sparse_pde.csv", problem_i,
                           complexity_i, degree_i, elapsed.count(), dimension,
                           numIterations, i, "Failed");
    }
    add_elapsed_time_to_vec(times, start, end);

    inform_user_about_kaczmarz_status(status);
  }

  return calc_avgtime(times);
}

///////////////////////////////////////////
// Helper functions
///////////////////////////////////////////

// Function to read the matrix from a file
SparseLinearSystem read_matrix_from_file(const std::string &file_path) {
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  const SparseLinearSystem lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  lse_input_stream.close();
  return lse;
}

// Read in the dimension of a file
unsigned int read_dimension(const std::string &file_path) {
  unsigned nnz, rows, cols;
  std::ifstream lse_input_stream(file_path);
  if (!lse_input_stream) {
    throw std::runtime_error("Failed to open matrix file: " + file_path);
  }
  lse_input_stream >> nnz >> rows >> cols;
  lse_input_stream.close();
  return rows;
}

// Function to generate the file path for a given problem, complexity, and
// degree
std::string generate_file_path(unsigned int problem, unsigned int complexity,
                               unsigned int degree) {
  return "../../generated_bvp_matrices/problem" + std::to_string(problem) +
         "/problem" + std::to_string(problem) + "_complexity" +
         std::to_string(complexity) + "_degree" + std::to_string(degree) +
         ".txt";
}

// Function to add elapsed time to a vector calculated from start and end time
// points
void add_elapsed_time_to_vec(
    std::vector<double> &times,
    const std::chrono::time_point<std::chrono::high_resolution_clock> start,
    const std::chrono::time_point<std::chrono::high_resolution_clock> end) {
  std::chrono::duration<double> elapsed = end - start;
  times.push_back(elapsed.count());
}

// This function writes the header to the file at the given path
void write_header(const std::string &file_path) {
  // Open the file in truncation mode to clear existing content
  std::ofstream outFile(
      file_path);  // Default mode is std::ios::out | std::ios::trunc
  if (outFile.is_open()) {
    outFile << "Problem,Complexity,Degree,Time,Dim,MaxIt,Precision,"
               "MaxComplexity,MaxDegree,NumIt,TimeThreshold,Iteration,Status\n";
    outFile.flush();  // Ensure data is written to the disk immediately
    outFile.close();
  } else {
    std::cerr << "Error: Unable to open file " << file_path << " for writing."
              << std::endl;
  }
}

// Utility function to write raw times to files
void write_result_to_file(const std::string &file_name, unsigned int problem,
                          unsigned int complexity, unsigned int degree,
                          double time, unsigned int dimension,
                          unsigned int num_it, unsigned int iteration,
                          const std::string &status) {
  std::ofstream outFile(file_name, std::ios::app);
  if (!outFile.is_open()) {
    std::cerr << "Error: Unable to open file " << file_name
              << " for writing.\n";
    return;
  }
  outFile << problem << "," << complexity << "," << degree << "," << time << ","
          << dimension << "," << MAX_IT << "," << PRECISION << ","
          << MAX_COMPLEXITY << "," << MAX_DEGREE << "," << num_it << ","
          << TIME_THRESHOLD << "," << iteration << "," << status << "\n";
  outFile.flush();  // Ensure data is written to the disk immediately
  outFile.close();

  // Alternatively only write into file if the status == "Converged", don't know
  // what is better, I would rather just write everything and then in the post
  // processing look at the status, but if you want to only store the converged
  // ones, uncomment this code and comment the one above

  // if (status == "Converged") {
  //   std::ofstream outFile(file_name, std::ios::app);
  //   if (!outFile.is_open()) {
  //     std::cerr << "Error: Unable to open file " << file_name
  //               << " for writing.\n";
  //     return;
  //   }
  //   outFile << problem << "," << complexity << "," << degree << "," << time
  //           << "," << dimension << "," << MAX_IT << "," << PRECISION << ","
  //           << MAX_COMPLEXITY << "," << MAX_DEGREE << "," << num_it << ","
  //           << TIME_THRESHOLD << "," << iteration << "," << status << "\n";
  //   outFile.flush();  // Ensure data is written to the disk immediately
  //   outFile.close();
  // }
}

// Calculates the avg time
double calc_avgtime(const std::vector<double> &times) {
  double avg_time = 0.0, std_dev = 0.0;
  compute_statistics(times, avg_time, std_dev);
  return avg_time;
}

// Inform user about kaczmarz solver status
void inform_user_about_kaczmarz_status(KaczmarzSolverStatus status) {
  if (status == KaczmarzSolverStatus::ZeroNormRow) {
    std::cout << "Zero norm row detected" << std::endl;
  } else if (status == KaczmarzSolverStatus::OutOfIterations) {
    std::cout << "Max iterations reached" << std::endl;
  }
}

/// @brief Computes the average and standard deviation of a vector of times.
/// @param times A vector of times recorded for benchmarking in seconds.
/// @param avgTime Reference to store the computed average time in seconds.
/// @param stdDev Reference to store the computed standard deviation.
void compute_statistics(const std::vector<double> &times, double &avgTime,
                        double &stdDev) {
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
