#include <chrono>
#include <iostream>
#include <random>
#include <fstream>

#include "../solvers/banded.hpp"
#include "linear_systems/sparse.hpp"

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
  constexpr unsigned dim = 200'000;
  constexpr unsigned bandwidth = 2;
  constexpr unsigned max_iterations = 1'000'000;
  constexpr double precision = 1e-7;

  std::mt19937 rng(21);
  const BandedLinearSystem lse =
      BandedLinearSystem::generate_random_regular(rng, dim, bandwidth);

  /*
  const auto eigen_start = hrclock::now();
  const Vector x_eigen = lse.to_sparse_system().eigen_solve();
  const auto eigen_end = hrclock::now();

  std::cout << "Eigen solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   eigen_end - eigen_start)
                   .count()
            << " milliseconds" << std::endl;
  */

  Vector x_kaczmarz = Vector::Zero(dim);

  std::vector<double> residuals_L1;
  std::vector<double> residuals_L2;
  std::vector<double> residuals_Linf;

  const auto kaczmarz_start = hrclock::now();
  /*const auto status =
      kaczmarz_banded_serial(lse, x_kaczmarz, max_iterations, precision);*/
  // const auto status = asynchronous_gpu(lse.to_sparse_system(), x_kaczmarz,
  // max_iterations, precision, 10);
  SerialInterleavedBandedSolver().run_iterations_with_residuals(lse, x_kaczmarz, residuals_L1, residuals_L2, residuals_Linf, 500'000);

  const auto kaczmarz_end = hrclock::now();

  std::cout << "Kaczmarz solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   kaczmarz_end - kaczmarz_start)
                   .count()
            << " milliseconds" << std::endl;
  /*   std::cout << "Kaczmarz solver status: " << kaczmarz_status_string(status)
              << std::endl; */

  const Vector r = lse.to_sparse_system().b() - lse.to_sparse_system().A() * x_kaczmarz;

  std::cout << "residual L1 norm = " << r.lpNorm<1>() << std::endl;
  std::cout << "residual L2 norm = " << r.lpNorm<2>() << std::endl;
  std::cout << "residual Linf norm = " << r.lpNorm<Eigen::Infinity>() << std::endl;


  /*
  const Vector error = x_kaczmarz - x_eigen;

  std::cout << "error norms:\n";
  std::cout << "L1 = " << error.lpNorm<1>() << "\n";
  std::cout << "L2 = " << error.lpNorm<2>() << "\n";
  std::cout << "L_inf = " << error.lpNorm<Eigen::Infinity>() << std::endl;
  */

    const auto write_residuals = [](const std::string& outfile, const std::string& residual_type, const std::vector<double>& residuals) {
    std::cout << "writing " << residual_type << " residuals to " << outfile << std::endl;
    std::ofstream ofs(outfile);
    for (const double r : residuals) {
      ofs << r << '\n';
    }
  };
  write_residuals("serial_interleaved_residuals_L1", "L1", residuals_L1);
  write_residuals("serial_interleaved_residuals_L2", "L2", residuals_L2);
  write_residuals("serial_interleaved_residuals_Linf", "Linf", residuals_Linf);

  std::cout << std::endl;
}
