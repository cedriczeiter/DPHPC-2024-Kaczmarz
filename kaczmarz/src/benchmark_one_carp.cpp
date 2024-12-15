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
#include "solvers/cusolver.hpp"

using hrclock = std::chrono::high_resolution_clock;

#define NRUNS 10

/**
 * The purpose of this file is to be easily able to manually benchmark a single
 * run of one of our implementations for the carp solver.
 */

/// Beware: For serious timed testing, we need to put up the L_RESIDUAL in
/// carp_utils.hpp up to maybe like 1000

int main() {
  std::cout << "  ____    _    ____  ____  " << std::endl;
  std::cout << " / ___|  / \\  |  _ \\|  _ \\ " << std::endl;
  std::cout << "| |     / _ \\ | |_) | |_) |" << std::endl;
  std::cout << "| |___ / ___ \\|  __/|  __/ " << std::endl;
  std::cout << " \\____/_/   \\_\\_|\\_\\|_|    " << std::endl;

  std::cout << "\n \n CARP solver benchmarking" << std::endl;

  // Read in the system from file
  std::ifstream lse_input_stream(
      "../../generated_bvp_matrices/problem1_complexity8_degree1.txt");
  const SparseLinearSystem sparse_lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

  // Define Variables
  const unsigned dim = sparse_lse.row_count();
  const unsigned max_iterations = std::numeric_limits<unsigned int>::max() - 1;

  std::cout << "Dimension: \n" << dim << std::endl;

  // Open file to write results to
  std::ofstream outFile("carp-cg.csv");
  outFile << "Precision,Carp,Eigen\n";  // Write the header for the CSV file

  //////////////////////////////////////////
  // Calculating the precise solution
  //////////////////////////////////////////

  // Calculate precise solution with Eigen non-iterative solver
  Vector x_precise = Vector::Zero(dim);
  const auto A = sparse_lse.A();
  const auto b = sparse_lse.b();

  const auto clock_start_eigen_non_it = hrclock::now();
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(A);
  x_precise = solver.solve(b);
  const auto clock_end_eigen_non_it = hrclock::now();
  const auto time_eigen_non_it =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          clock_end_eigen_non_it - clock_start_eigen_non_it)
          .count();

  std::cout << "Eigen non-iterative solution computed in " << time_eigen_non_it
            << " milliseconds \n \n -------------- \n \n"
            << std::endl;

  double precision = 1;  // precision gets multiplied by 0.1 in each iteration
  for (int i = 0; i < NRUNS; i++) {
    //////////////////////////////////////////
    // Calculating the solution with CARP
    //////////////////////////////////////////
    Vector x_kaczmarz = Vector::Zero(dim);

    const auto kaczmarz_start = hrclock::now();
    int nr_of_steps =
        0;  // just a placeholder, used in benchmark_one_carp_lambda.cpp
    const double relaxation =
        0.35;  // this relaxation param was found empirically by
               // benchmark_one_carp_lambda_auto; there is no guarantee of
               // convergence! if solver doesnt converge, change relaxation to
               // 1/max_nnz_per_col
    const auto status = carp_gpu(sparse_lse, x_kaczmarz, max_iterations,
                                 precision, relaxation, nr_of_steps);
    const auto kaczmarz_end = hrclock::now();

    const auto kaczmarz_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(kaczmarz_end -
                                                              kaczmarz_start)
            .count();
    if (status == KaczmarzSolverStatus::ZeroNormRow) {
      std::cout << "Zero norm row detected" << std::endl;
    } else if (status == KaczmarzSolverStatus::OutOfIterations) {
      std::cout << "Max iterations reached" << std::endl;
    } else {
      std::cout << "Kaczmarz solution converged in " << kaczmarz_time
                << " milliseconds for precision " << precision << std::endl;
    }

    //////////////////////////////////////////
    // Calculating the solution with Eigen iterative solver
    //////////////////////////////////////////

    Vector x_iter = Vector::Zero(dim);
    const auto iter_start = hrclock::now();
    const auto A = sparse_lse.A();
    const auto b = sparse_lse.b();
    Eigen::BiCGSTAB<SparseMatrix> lscg(A);
    lscg.setTolerance(precision);
    lscg.setMaxIterations(max_iterations);
    x_iter = lscg.solve(b);
    const auto iter_end = hrclock::now();
    /*const auto iter_start = hrclock::now();
    cusolver(sparse_lse, x_iter, max_iterations, precision);
    const auto iter_end = hrclock::now();*/
    const auto iter_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(iter_end -
                                                              iter_start)
            .count();

    std::cout << "Iterative solution computed in " << iter_time
              << " milliseconds for precision " << precision << std::endl;

    //////////////////////////////////////////
    // Write and print both results
    //////////////////////////////////////////

    std::cout << "Precision: " << precision << "\nTime CARP / Time Eigen: "
              << (double)kaczmarz_time / (double)iter_time << "\n"
              << std::endl;

    // write to csv
    outFile << precision << "," << kaczmarz_time << "," << iter_time
            << std::endl;
    precision = precision * 0.1;

    // Print errors compared to non-iterative solver in L1 and Linf norm
    std::cout
        << "Error compared to non-iterative solver in L1 norm for Kaczmarz: "
        << (x_kaczmarz - x_precise).lpNorm<1>() << std::endl;
    std::cout << "Error compared to non-iterative solver in Linf norm: "
              << (x_kaczmarz - x_precise).lpNorm<Eigen::Infinity>()
              << std::endl;

    std::cout << "Error compared to non-iterative solver in L1 norm for Eigen "
                 "iterative: "
              << (x_iter - x_precise).lpNorm<1>() << std::endl;
    std::cout << "Error compared to non-iterative solver in Linf norm: "
              << (x_iter - x_precise).lpNorm<Eigen::Infinity>() << std::endl;

    // Print the solution vectors
    std::cout << "print first 40 elements of each results vector: "
              << std::endl;
    std::cout << "Eigen: " << std::endl;
    for (int i = 0; i < 40; i++) {
      std::cout << x_precise[i] << "   ";
    }
    std::cout << "\nKaczmarz: " << std::endl;
    for (int i = 0; i < 40; i++) {
      std::cout << x_kaczmarz[i] << "   ";
    }
    std::cout << "\nIterative: " << std::endl;
    for (int i = 0; i < 40; i++) {
      std::cout << x_iter[i] << "   ";
    }
    std::cout << std::endl;

    std::cout << "----------------------------------- \n \n" << std::endl;
  }
}
