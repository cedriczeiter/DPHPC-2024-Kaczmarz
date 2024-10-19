#include "random_dense_system.hpp"

#include <Eigen/Dense>

void generate_random_dense_linear_system(double* A, double* b, double* x, unsigned dim){//seeding should be done once in main function!
  
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_m(A, dim, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> b_m(b, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> x_m(x, dim);
  
  do {
    for (int i = 0; i < dim; i++){
      for (int j = 0; j < dim; j++){
         A_m(i,j) = static_cast<double>(std::rand())/RAND_MAX;
      }
      b_m(i) = static_cast<double>(std::rand())/RAND_MAX;
    }
  } while (A_m.fullPivLu().rank() != dim); // check if matrix is full-rank (will be full-rank practically always, but still)
  
  x_m = A_m.fullPivLu().solve(b_m);
}
