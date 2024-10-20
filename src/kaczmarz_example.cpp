#include <iostream>
#include <random>

#include "kaczmarz.hpp"
#include "linear_systems/dense.hpp"

// this is an example, demonstrating how to call the kaczmarz solver (serial
// version) on a randomly generated dense matrix

// Rather refer to the actual testing code instead, slight changes there!

int main() {
  const unsigned dim = 5;

  std::mt19937 rng(21);
  const DenseLinearSystem lse =
      DenseLinearSystem::generate_random_regular(rng, dim);

  std::vector<double> x_kaczmarz(dim, 0.0);

  const auto status = dense_kaczmarz(lse, &x_kaczmarz[0], 100000, 1e-10);
  if (status != KaczmarzSolverStatus::Converged) {
    std::cout << "The Kaczmarz solver didn't converge!" << std::endl;
  }

  std::cout << "Kaczmarz solution: \n";
  for (unsigned i = 0; i < dim; i++) {
    std::cout << x_kaczmarz[i] << std::endl;
  }

  std::cout << "\n\n";

  const Vector x_eigen = lse.eigen_solve();

  std::cout << "Eigen solution: \n";
  for (unsigned i = 0; i < dim; i++) {
    std::cout << x_eigen[i] << std::endl;
  }
}
