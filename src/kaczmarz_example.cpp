#include "random_dense_system.hpp"
#include "kaczmarz.hpp"

#include <iostream>
#include <random>

//this is an example, demonstrating how to call the kaczmarz solver (serial version) on a randomly generated dense matrix


int main() {
  int dim = 5;
  double *A = (double *)malloc(sizeof(double)*dim*dim);
  double *b = (double *)malloc(sizeof(double)*dim);
  double *x = (double *)malloc(sizeof(double)*dim);
  
  std::mt19937 rng(21);
  generate_random_dense_linear_system(rng, A, b, x, dim);


  double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);

  const auto status = kaczmarz_solver(A, b, x_kaczmarz, dim, dim, 100000, 1e-10);
  if (status != KaczmarzSolverStatus::Converged) {
    std::cout << "The Kaczmarz solver didn't converge!" << std::endl;
  }

  std::cout << "Eigen solution: \n";
  for (int i = 0; i < dim; i++){
    std::cout << x[i] << std::endl;
  }
  std::cout << "\n\n";

  std::cout << "Kaczmarz solution: \n";
  for (int i = 0; i < dim; i++){
    std::cout << x_kaczmarz[i] << std::endl;
  }
}
