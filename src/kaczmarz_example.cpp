#include "random_dense_system.hpp"
#include <iostream>
extern "C" { 
  #include "kaczmarz.h" 
} 

//this is an example, demonstrating how to call the kaczmarz solver (serial version) on a randomly generated dense matrix


int main() {
  int dim = 5;
  double *A = (double *)malloc(sizeof(double)*dim*dim);
  double *b = (double *)malloc(sizeof(double)*dim);
  double *x = (double *)malloc(sizeof(double)*dim);
  
  std::srand(21);
  get_dense_linear_system(A, b, x, dim);


  double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);

  kaczmarz_solver(A, b, x_kaczmarz, dim, dim, 100000, 1e-10);

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
