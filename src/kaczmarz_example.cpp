#include "kaczmarz_serial/kaczmarz_common.hpp"
#include "kaczmarz_serial/random_kaczmarz.hpp"
#include "linear_systems/dense.hpp"
#include "kaczmarz.hpp"
#include "random_kaczmarz.hpp"

#include <iostream>
#include <iterator>
#include <random>

int main() {
  const unsigned dim = 5;
  
  // Initialize the random number generator
  std::mt19937 rng(21);

  const DenseLinearSystem lse =  DenseLinearSystem::generate_random_regular(rng, dim);

  std::vector<double> x_kaczmarz(dim, 0.0);
  std::vector<double> x_kaczmarz_random(dim, 0.0);

  const auto status_dense = dense_kaczmarz(lse, &x_kaczmarz[0], 100000, 1e-10);
  if (status_dense != KaczmarzSolverStatus::Converged) {
    std::cout << "The serial Kaczmarz solver didn't converge!" << std::endl;
  }

  const auto status_random = kaczmarz_random_solver(lse, &x_kaczmarz_random[0], 100000, 1e-10);
  if (status_random != KaczmarzSolverStatus::Converged){
    std::cout << "The random Kaczmarz solver didn't converge!" << std::endl;
  }

  std::cout << "Kaczmarz solution: \n";
  for (unsigned i = 0; i < dim; i++){
    std::cout << x_kaczmarz[i] << std::endl;
  }
  std::cout << "\n\n";

  // Print the solution from the randomized Kaczmarz solver
  std::cout << "Randomized Kaczmarz solution: \n";
  for (int i = 0; i < dim; i++){
    std::cout << x_kaczmarz_random[i] << std::endl;
  }

  std::cout << "\n\n";

  const Vector x_eigen = lse.eigen_solve();

  std::cout << "Eigen solution: \n";
  for (unsigned i = 0; i < dim; i++){
    std::cout << x_eigen[i] << std::endl;
  }
}
