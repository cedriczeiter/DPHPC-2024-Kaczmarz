#include "random_dense_system.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <random>

void generate_random_dense_linear_system(std::mt19937& rng, double* A, double* b, double* x, unsigned dim){
  
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A_m(A, dim, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> b_m(b, dim);
  Eigen::Map<Eigen::Vector<double, Eigen::Dynamic>> x_m(x, dim);

  std::uniform_real_distribution<> dist(-1.0, 1.0);
  const auto generate_element = [&dist, &rng]() { return dist(rng); };
  
  do {
    std::generate_n(A, dim * dim, generate_element);
    std::generate_n(b, dim, generate_element);
  } while (A_m.fullPivLu().rank() != dim); // check if matrix is full-rank (will be full-rank practically always, but still)
  
  x_m = A_m.fullPivLu().solve(b_m);
}
