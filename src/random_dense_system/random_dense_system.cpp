#include "random_dense_system.hpp"
#include <Eigen/Dense>
#include <iostream>


void get_dense_linear_system(double* A, double* b, double* x, unsigned dim){//seeding should be done once in main function!
  
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_m(A, dim, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> b_m(b, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> x_m(x, dim);
  

  unsigned full_rank = 0;
  
  while (!full_rank){
    for (int i = 0; i < dim; i++){
      for (int j = 0; j < dim; j++){
         A_m(i,j) = static_cast<double>(std::rand())/RAND_MAX;
      }
      b_m(i) = static_cast<double>(std::rand())/RAND_MAX;
    }
    if (A_m.fullPivLu().rank() == dim) full_rank = 1; //check if matrix is full-rank
  }
  x_m = A_m.fullPivLu().solve(b_m);
  //std::cout << A_m << std::endl;
  /*A = A_m.data();
  b = b_m.data();
  x = x_m.data();*/
}
