#include <Eigen/Sparse>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

#include "carp.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

using hrclock = std::chrono::high_resolution_clock;

#define PRECISION 1e-9
#define L_RESIDUAL 1

/**
 * The purpose of this file is to be easily able to manually benchmark a single
 * run of one of our implementations for the carp solver.
 */

int main() {
  std::cout << "  ____    _    ____  ____  " << std::endl;
  std::cout << " / ___|  / \\  |  _ \\|  _ \\ " << std::endl;
  std::cout << "| |     / _ \\ | |_) | |_) |" << std::endl;
  std::cout << "| |___ / ___ \\|  __/|  __/ " << std::endl;
  std::cout << " \\____/_/   \\_\\_|\\_\\|_|    " << std::endl;

  std::cout << "\n \n CARP find relaxation parameter auto on all files"
            << std::endl;

  std::string file_path = "../../generated_bvp_matrices";

  double end_relaxation = 0.7;
  double step_relaxation = 0.05;

  const unsigned max_iterations =
      40000;  // set such that it doesnt take tooo long

  // Files are in file_path inside of three problem folders called problem1,
  // problem2, problem3

  std::filesystem::path path = file_path;
  std::filesystem::recursive_directory_iterator files(path);

  for (const auto &entry : files) {
    // file ends by txt and begins with problem and does not end by banded.txt
    if (entry.path().extension() == ".txt" &&
        entry.path().filename().string().substr(0, 7) == "problem" &&
        entry.path().filename().string().substr(
            entry.path().filename().string().size() - 10) != "banded.txt") {
      std::cout << "----------------------------------- \n" << std::endl;
      std::cout << "in file: " << entry.path() << std::endl;

      // Read in the system from file
      std::ifstream lse_input_stream(entry.path());

      const SparseLinearSystem sparse_lse =
          SparseLinearSystem::read_from_stream(lse_input_stream);
      // Define Variables
      const unsigned dim = sparse_lse.row_count();
      // const unsigned max_iterations = std::numeric_limits<unsigned
      // int>::max() - 1;

      std::cout << "Dimension: \n" << dim << std::endl;

      // Open file to write results to
      std::string file_name = entry.path().filename().string();
      std::ofstream outFile(
          "../../generated_bvp_matrices/lambda_experiments_new/"
          "carp-cg-lambda-steps-" +
          file_name + ".csv");  // overwrites file if it already exists
      outFile
          << "Relaxation,Carp_steps\n";  // Write the header for the CSV file

      double start_relaxation = 0.2;

      while (start_relaxation < end_relaxation) {
        std::cout << "----------------------------------- \n" << std::endl;
        std::cout << "Relaxation: " << start_relaxation << " out of "
                  << end_relaxation << std::endl;

        Vector x_kaczmarz = Vector::Zero(dim);

        int nr_of_steps = 0;
        const auto status = carp_gpu(sparse_lse, x_kaczmarz, max_iterations,
                                     PRECISION, start_relaxation, nr_of_steps);

        // Print the status of the Kaczmarz solver to terminal
        if (status == KaczmarzSolverStatus::ZeroNormRow) {
          std::cout << "Zero norm row detected" << std::endl;
        } else if (status == KaczmarzSolverStatus::OutOfIterations) {
          std::cout << "Max iterations reached" << std::endl;
          break;
        } else {
          std::cout << " --- Relaxation: " << start_relaxation
                    << " --- Nr. of steps: " << nr_of_steps << std::endl;
        }

        // write to csv
        outFile << start_relaxation << "," << nr_of_steps << "\n";

        start_relaxation += step_relaxation;
      }
    }
  }

  std::cout << "----------------------------------- \n \n" << std::endl;
}
