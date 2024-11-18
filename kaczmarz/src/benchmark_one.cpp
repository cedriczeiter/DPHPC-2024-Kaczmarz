#include <chrono>
#include <iostream>
#include <random>

#include "linear_systems/dense.hpp"
#include "solvers/asynchronous.hpp"
#include "solvers/banded.hpp"
#include "solvers/basic_cuda.hpp"

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
  constexpr unsigned dim = 50;
  constexpr unsigned max_iterations = 10000;
  constexpr double precision = 1e-7;

  std::mt19937 rng(21);
  const DenseLinearSystem lse =
      DenseLinearSystem::generate_random_regular(rng, dim);

  const auto eigen_start = hrclock::now();
  const Vector x_eigen = lse.eigen_solve();
  const auto eigen_end = hrclock::now();

  std::cout << "Eigen solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   eigen_end - eigen_start)
                   .count()
            << " milliseconds" << std::endl;

  Vector x_kaczmarz = Vector::Zero(dim);

  const auto kaczmarz_start = hrclock::now();

    std::vector<double> times_residuals;
  std::vector<double> residuals;
  std::vector<int> iterations;
  const int convergence_step_rate = 10000;

  const auto status = dense_kaczmarz_cuda(lse, x_kaczmarz.data(), max_iterations, precision, times_residuals, residuals, iterations, convergence_step_rate);
  const auto kaczmarz_end = hrclock::now();

  std::cout << "Kaczmarz solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   kaczmarz_end - kaczmarz_start)
                   .count()
            << " milliseconds" << std::endl;
  std::cout << "Kaczmarz solver status: " << kaczmarz_status_string(status)
            << std::endl;

  const Vector error = x_kaczmarz - x_eigen;

  std::cout << "error norms:\n";
  std::cout << "L1 = " << error.lpNorm<1>() << "\n";
  std::cout << "L_inf = " << error.lpNorm<Eigen::Infinity>() << std::endl;


 std::cout << "Eigen: " <<std::endl;
  for (int i = 0; i < dim; i++){
    std::cout << x_eigen[i] << "   ";
  }
  std::cout << "\nKaczmarz: " << std::endl;
  for (int i = 0; i < dim; i++){
    std::cout << x_kaczmarz[i] << "   ";
  }
  std::cout << std::endl;
}
