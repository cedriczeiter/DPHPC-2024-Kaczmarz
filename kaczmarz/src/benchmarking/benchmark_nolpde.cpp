#include <chrono>
#include <fstream>
#include <iostream>

#include "linear_systems/discretization.hpp"
#include "../solvers/nolpde.hpp"

using hrclock = std::chrono::high_resolution_clock;

int main() {
  const Discretization d = []() {
    std::ifstream ifs(
        "../../generated_bvp_matrices/problem3/"
        "problem3_complexity8_degree1.txt");
    return Discretization::read_from_stream(ifs);
  }();

  std::cout << "dim = " << d.position_hints.size() << std::endl;

  Vector x_kaczmarz = Vector::Zero(d.sys.A().cols());

  std::vector<double> residuals_L1;
  std::vector<double> residuals_L2;
  std::vector<double> residuals_Linf;

  const auto kaczmarz_start = hrclock::now();
  //CUDANolPDESolver(16, 128).run_iterations_with_residuals(d, x_kaczmarz, residuals_L1, residuals_L2, residuals_Linf, 3000);
  //PermutingSerialNolPDESolver(16 * 128).run_iterations_with_residuals(d, x_kaczmarz, residuals_L1, residuals_L2, residuals_Linf, 30'000);
  //ShuffleSerialNolPDESolver(321).run_iterations(d, x_kaczmarz, 1000);
  BasicSerialNolPDESolver().run_iterations_with_residuals(d, x_kaczmarz, residuals_L1, residuals_L2, residuals_Linf, 30'000);
  //ShuffleSerialNolPDESolver(321).run_iterations_with_residuals(d, x_kaczmarz, residuals_L1, residuals_L2, residuals_Linf, 30'000);
  const auto kaczmarz_end = hrclock::now();

  const Vector r = d.sys.b() - d.sys.A() * x_kaczmarz;

  std::cout << "residual L1 norm = " << r.lpNorm<1>() << std::endl;
  std::cout << "residual L2 norm = " << r.lpNorm<2>() << std::endl;
  std::cout << "residual Linf norm = " << r.lpNorm<Eigen::Infinity>() << std::endl;

  std::cout << "Kaczmarz solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   kaczmarz_end - kaczmarz_start)
                   .count()
            << " milliseconds" << std::endl;

  const auto write_residuals = [](const std::string& outfile, const std::string& residual_type, const std::vector<double>& residuals) {
    std::cout << "writing " << residual_type << " residuals to " << outfile << std::endl;
    std::ofstream ofs(outfile);
    for (const double r : residuals) {
      ofs << r << '\n';
    }
  };
  write_residuals("basic_residuals_L1", "L1", residuals_L1);
  write_residuals("basic_residuals_L2", "L2", residuals_L2);
  write_residuals("basic_residuals_Linf", "Linf", residuals_Linf);
}
