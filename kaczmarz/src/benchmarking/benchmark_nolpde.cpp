#include <chrono>
#include <fstream>
#include <iostream>

#include "linear_systems/discretization.hpp"
#include "nolpde.hpp"

using hrclock = std::chrono::high_resolution_clock;

int main() {
  const Discretization d = []() {
    std::ifstream ifs("../TODO_give_actual_name.txt");
    return Discretization::read_from_stream(ifs);
  }();

  Vector x_kaczmarz(d.sys.A().cols());

  const auto kaczmarz_start = hrclock::now();
  CUDANolPDESolver(16, 128).run_iterations(d, x_kaczmarz, 100'000);
  const auto kaczmarz_end = hrclock::now();

  std::cout << "Kaczmarz solution computed in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   kaczmarz_end - kaczmarz_start)
                   .count()
            << " milliseconds" << std::endl;
}
