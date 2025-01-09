#include <chrono>
#include <fstream>
#include <iostream>

#include "linear_systems/discretization.hpp"
#include "nolpde.hpp"

using hrclock = std::chrono::high_resolution_clock;

int main() {
  const Discretization d = []() {
    std::ifstream ifs(
        "../../generated_bvp_matrices/problem3/"
        "problem3_complexity8_degree1.txt");
    return Discretization::read_from_stream(ifs);
  }();

  std::cout << "dim = " << d.position_hints.size() << std::endl;

  Vector x_kaczmarz(d.sys.A().cols());

  const auto kaczmarz_start = hrclock::now();
  // CUDANolPDESolver(16, 128).run_iterations(d, x_kaczmarz, 2000);
  // PermutingSerialNolPDESolver(16 * 128).run_iterations(d, x_kaczmarz, 2000);
  // BasicSerialNolPDESolver().run_iterations(d, x_kaczmarz, 1000);
  ShuffleSerialNolPDESolver(321).run_iterations(d, x_kaczmarz, 1000);
  const auto kaczmarz_end = hrclock::now();

  const Vector r = d.sys.b() - d.sys.A() * x_kaczmarz;

  std::cout << "residual L2 norm = " << r.lpNorm<2>() << std::endl;

  std::cout << "Kaczmarz solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   kaczmarz_end - kaczmarz_start)
                   .count()
            << " milliseconds" << std::endl;
}
