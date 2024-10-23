#include <iostream>
#include <random>

#include "kaczmarz.hpp"
#include "random_dense_system.hpp"
#include "random_kaczmarz.hpp"




/* test    afds asdfasdf  asdf asdfasd fa sdfasdf asfasf asd f fa dsfa dsaf asdf adsf fasdf sadf sda af  



*/

int main() {
  // Initialize the space that we need.
  int dim = 5;
  double *A = (double *)malloc(sizeof(double) * dim * dim);
  double *b = (double *)malloc(sizeof(double) * dim);
  double *x = (double *)malloc(sizeof(double) * dim);

  // Initialize the random number generator
  std::mt19937 rng(21);

  // Generate a random dense linear system
  generate_random_dense_linear_system(rng, A, b, x, dim);

  // Allocate space for the solutions
  double *x_kaczmarz = (double *)malloc(sizeof(double) * dim);
  double *x_kaczmarz_random = (double *)malloc(sizeof(double) * dim);

  // Test the serial Kaczmarz solver
  const auto status =
      kaczmarz_solver(A, b, x_kaczmarz, dim, dim, 100000, 1e-10);
  if (status != KaczmarzSolverStatus::Converged) {
    std::cout << "The serial Kaczmarz solver didn't converge!" << std::endl;
  }

  // Test the randomized Kaczmarz solver
  const auto random_status =
      kaczmarz_random_solver(A, b, x_kaczmarz_random, dim, dim, 100000, 1e-10);
  if (random_status != KaczmarzSolverStatus::Converged) {
    std::cout << "The randomized Kaczmarz solver didn't converge!" << std::endl;
  }

  // Print the original solution from Eigen
  std::cout << "Eigen solution: \n";
  for (int i = 0; i < dim; i++) {
    std::cout << x[i] << std::endl;
  }
  std::cout << "\n\n";

  // Print the solution from the serial Kaczmarz solver
  std::cout << "Serial Kaczmarz solution: \n";
  for (int i = 0; i < dim; i++) {
    std::cout << x_kaczmarz[i] << std::endl;
  }
  std::cout << "\n\n";

  // Print the solution from the randomized Kaczmarz solver
  std::cout << "Randomized Kaczmarz solution: \n";
  for (int i = 0; i < dim; i++) {
    std::cout << x_kaczmarz_random[i] << std::endl;
  }

  // Free the allocated memory
  free(A);
  free(b);
  free(x);
  free(x_kaczmarz);
  free(x_kaczmarz_random);
}
