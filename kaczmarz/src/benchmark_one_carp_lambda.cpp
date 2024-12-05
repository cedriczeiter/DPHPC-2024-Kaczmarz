#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>

#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"
#include "solvers/asynchronous.hpp"
#include "solvers/banded.hpp"
#include "solvers/carp.hpp"

using hrclock = std::chrono::high_resolution_clock;

#define PRECISION 1e-10
#define NRUNS 1  // usefull if we want to time the stuff multiple times

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

  std::cout << "\n \n CARP find relaxation parameter" << std::endl;

  // Read in the system from file
  std::ifstream lse_input_stream(
      "../../generated_bvp_matrices/problem1_complexity3_degree3.txt");
  const SparseLinearSystem sparse_lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

  // Define Variables
  const unsigned dim = sparse_lse.row_count();
  // const unsigned max_iterations = std::numeric_limits<unsigned int>::max() -
  // 1;
  const unsigned max_iterations =
      100'000;  // set such that it doesnt take tooo long

  std::cout << "Dimension: \n" << dim << std::endl;

  // Open file to write results to
  std::ofstream outFile("carp-cg-lambda-relax-time.csv");
  outFile << "Relaxation,Carp_time\n";  // Write the header for the CSV file

  std::ofstream outFile2("carp-cg-lambda-relax-steps-p1-7.csv");
  outFile2 << "Relaxation,Carp_steps\n";  // Write the header for the CSV file

  double start_relaxation = 0.1;
  double end_relaxation = 30.0;
  double step_relaxation = 0.20;

  while (start_relaxation < end_relaxation) {
    std::cout << "----------------------------------- \n" << std::endl;
    std::cout << "Relaxation: " << start_relaxation << " out of "
              << end_relaxation << std::endl;

    // let each relaxation parameter run for NRUNS times
    for (int i = 0; i < NRUNS; i++) {
      Vector x_kaczmarz = Vector::Zero(dim);

      // Timing the Kaczmarz solver
      const auto kaczmarz_start = hrclock::now();
      int nr_of_steps = 0;
      const auto status = carp_gpu(sparse_lse, x_kaczmarz, max_iterations,
                                   PRECISION, start_relaxation, nr_of_steps);
      const auto kaczmarz_end = hrclock::now();

      const auto kaczmarz_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(kaczmarz_end -
                                                                kaczmarz_start)
              .count();

      // Print the status of the Kaczmarz solver to terminal
      if (status == KaczmarzSolverStatus::ZeroNormRow) {
        std::cout << "Zero norm row detected" << std::endl;
      } else if (status == KaczmarzSolverStatus::OutOfIterations) {
        std::cout << "Max iterations reached" << std::endl;
      } else {
        int nr_runs = NRUNS;
        std::cout << "Time: " << kaczmarz_time
                  << " milliseconds --- Relaxation: " << start_relaxation
                  << " --- Nr. of steps: " << nr_of_steps
                  << "   >>>> Internal run: " << i << "/" << nr_runs
                  << std::endl;
      }

      // write to csv
      outFile << start_relaxation << "," << kaczmarz_time << "\n";
      outFile2 << start_relaxation << "," << nr_of_steps << "\n";
    }

    start_relaxation += step_relaxation;
  }

  std::cout << "----------------------------------- \n \n" << std::endl;
}
