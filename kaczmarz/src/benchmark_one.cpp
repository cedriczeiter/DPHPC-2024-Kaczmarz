#include <chrono>
#include <random>
#include <iostream>

#include "linear_systems/sparse.hpp"
#include "kaczmarz_parallel/kaczmarz_banded.hpp"

using hrclock = std::chrono::high_resolution_clock;

int main() {
  constexpr unsigned dim = 2000;
  constexpr unsigned bandwidth = 6;
  constexpr unsigned max_iterations = 1'000'000'000;
  constexpr double precision = 1e-5;

  std::mt19937 rng(21);
  const BandedLinearSystem lse =
    BandedLinearSystem::generate_random_regular(rng, dim, bandwidth);

  const auto eigen_start = hrclock::now();
  const Vector x_eigen = lse.to_sparse_system().eigen_solve();
  const auto eigen_end = hrclock::now();

  std::cout << "Eigen solution computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(eigen_end - eigen_start) << std::endl;

  Vector x_kaczmarz = Vector::Zero(dim);

  const auto kaczmarz_start = hrclock::now();
  const auto status = kaczmarz_banded_2_cpu_threads(lse, x_kaczmarz, max_iterations, precision);
  const auto kaczmarz_end = hrclock::now();

  std::cout << "Kaczmarz solution computed in " << std::chrono::duration_cast<std::chrono::milliseconds>(kaczmarz_end - kaczmarz_start) << std::endl;
  std::cout << "Kaczmarz solver status: " << kaczmarz_status_string(status) << std::endl;

  const Vector error = x_kaczmarz - x_eigen;

  std::cout << "error norms:\n";
  std::cout << "L1 = " << error.lpNorm<1>() << "\n";
  std::cout << "L_inf = " << error.lpNorm<Eigen::Infinity>() << std::endl;;
}
