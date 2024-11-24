#include <Eigen/IterativeLinearSolvers>
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

/**
 * The purpose of this file is to be easily able to manually benchmark a single
 * run of one of our implementations.
 *
 * This is useful because some performance characteristics might only be
 * visible for large matrices. For those, it is impractically long to run the
 * other benchmarking which repeats execution multiple times.
 */

int main() {
  // constexpr unsigned dim = 5000;
  constexpr unsigned bandwidth = 2;
  // constexpr unsigned max_iterations = 100'000;
  // constexpr double precision = 1e-1;

  /*std::mt19937 rng(13);
  const auto sparse_lse =
      BandedLinearSystem::generate_random_regular(rng, dim,
  bandwidth).to_sparse_system();*/

  std::ifstream lse_input_stream(
      "../../generated_bvp_matrices/problem1_complexity6.txt");
  const SparseLinearSystem sparse_lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);

  const unsigned dim = sparse_lse.row_count();

  std::cout << "Dimension: " << dim << std::endl;

  /*const auto eigen_start = hrclock::now();
  const Vector x_eigen = lse.to_sparse_system().eigen_solve();
  const auto eigen_end = hrclock::now();

  std::cout << "Eigen solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   eigen_end - eigen_start)
                   .count()
            << " milliseconds" << std::endl;*/

  double precision = 0.5;
  for (int i = 0; i < 20; i++) {
    const unsigned max_iterations =
        std::numeric_limits<unsigned int>::max() - 1;
    Vector x_kaczmarz = Vector::Zero(dim);

    const auto kaczmarz_start = hrclock::now();
    const auto status =
        carp_gpu(sparse_lse, x_kaczmarz, max_iterations, precision);
    const auto kaczmarz_end = hrclock::now();

    std::cout << "Kaczmarz solution computed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     kaczmarz_end - kaczmarz_start)
                     .count()
              << " milliseconds" << std::endl;

    Vector x_iter = Vector::Zero(dim);
    const auto iter_start = hrclock::now();
    const auto A = sparse_lse.A();
    const auto b = sparse_lse.b();
    Eigen::LeastSquaresConjugateGradient<SparseMatrix,
                                         Eigen::IdentityPreconditioner>
        lscg(A);
    // lscg.preconditioner() = Eigen::IdentityPreconditioner;
    lscg.setTolerance(precision);
    lscg.setMaxIterations(max_iterations);
    x_iter = lscg.solve(b);
    const auto iter_end = hrclock::now();

    std::cout << "Iterative solution computed in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     iter_end - iter_start)
                     .count()
              << " milliseconds" << std::endl;

    std::cout << "Precision: " << precision << "\nTime CARP / Time Eigen: "
              << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                     kaczmarz_end - kaczmarz_start)
                         .count() /
                     (double)
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             iter_end - iter_start)
                             .count()
              << "\n\n"
              << std::endl;
    precision = precision * 0.5;
  }
  /*std::cout << "Kaczmarz solver status: " << kaczmarz_status_string(status)
            << std::endl;*/

  // const auto banded_start = hrclock::now();
  /*const auto status =
      kaczmarz_banded_serial(lse, x_kaczmarz, max_iterations, precision);*/
  /*const auto status_banded =
      kaczmarz_banded_cuda(lse, x_kaczmarz, max_iterations, precision);
  const auto banded_end = hrclock::now();*/

  /*std::cout << "Banded solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   banded_end - banded_start)
                   .count()
            << " milliseconds" << std::endl;
  std::cout << "Banded solver status: " << kaczmarz_status_string(status)
            << std::endl;*/

  /*const Vector error = x_kaczmarz - x_eigen;

  std::cout << "error norms:\n";
  std::cout << "L1 = " << error.lpNorm<1>() << "\n";
  std::cout << "L_inf = " << error.lpNorm<Eigen::Infinity>() << std::endl;*/

  /*std::cout << "Eigen: " <<std::endl;
  for (int i = 0; i < dim; i++){
    std::cout << x_eigen[i] << "   ";
  }
  std::cout << "\nKaczmarz: " << std::endl;
  for (int i = 0; i < dim; i++){
    std::cout << x_kaczmarz[i] << "   ";
  }
  std::cout << std::endl;*/
}
