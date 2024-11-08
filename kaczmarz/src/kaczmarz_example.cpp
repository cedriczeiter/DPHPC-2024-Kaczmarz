#include <fstream>
#include <iostream>
#include <random>

#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "solvers/basic.hpp"
#include "solvers/common.hpp"
#include "solvers/random.hpp"
#include "solvers/basic_cuda.hpp"

int main() {
  const unsigned dim = 5;

  // Initialize the random number generator

  std::mt19937 rng(21);

  const DenseLinearSystem lse =
      DenseLinearSystem::generate_random_regular(rng, dim);

  std::vector<double> x_kaczmarz(dim, 0.0);
  std::vector<double> x_kaczmarz_random(dim, 0.0);
  std::vector<double> times_residuals;
  std::vector<double> residuals;
  std::vector<int> iterations;
  const auto status_dense =
      dense_kaczmarz(lse, &x_kaczmarz[0], 100000, 1e-10, times_residuals,
                     residuals, iterations, 100000);
  if (status_dense != KaczmarzSolverStatus::Converged) {
    std::cout << "The serial Kaczmarz solver didn't converge!" << std::endl;
  }

  std::cout << "Kaczmarz solution: \n";
  for (unsigned i = 0; i < dim; i++) {
    std::cout << x_kaczmarz[i] << std::endl;
  }
  std::cout << "\n\n";

  const auto status_random =
      kaczmarz_random_solver(lse, &x_kaczmarz_random[0], 100000, 1e-10,
                             times_residuals, residuals, iterations, 100000);
  if (status_random != KaczmarzSolverStatus::Converged) {
    std::cout << "The random Kaczmarz solver didn't converge!" << std::endl;
  }

  // Print the solution from the randomized Kaczmarz solver
  std::cout << "Randomized Kaczmarz solution: \n";
  for (int i = 0; i < dim; i++) {
    std::cout << x_kaczmarz_random[i] << std::endl;
  }

  std::cout << "\n\n";

  const Vector x_eigen = lse.eigen_solve();

  std::cout << "Eigen solution: \n";
  for (unsigned i = 0; i < dim; i++) {
    std::cout << x_eigen[i] << std::endl;
  }

  // adding example for sparse matrices
  std::cout << "\n\nNow solving for a sparse matrix generated from a mesh..."
            << std::endl;

  // read in from file
  std::ifstream lse_input_stream(
      "../../generated_bvp_matrices/elementmatrix_unitsquare.txt");
  const SparseLinearSystem sparse_lse =
      SparseLinearSystem::read_from_stream(lse_input_stream);
  // solve
  Eigen::VectorXd x_kaczmarz_sparse =
      Eigen::VectorXd::Zero(sparse_lse.column_count());
  const auto status_sparse =
      sparse_kaczmarz(sparse_lse, x_kaczmarz_sparse, 10000000, 1e-10,
                      times_residuals, residuals, iterations, 100000);
  if (status_sparse != KaczmarzSolverStatus::Converged) {
    std::cout
        << "The serial Kaczmarz solver for sparse matrices didn't converge!"
        << std::endl;
  }

  const unsigned cols = sparse_lse.column_count();

  const Vector x_eigen_sparse = sparse_lse.eigen_solve();

  std::cout << "\n\nSparse Kaczmarz solution: \n";
  for (int i = 0; i < cols; i++) {
    std::cout << x_kaczmarz_sparse[i] << std::endl;
  }

  std::cout << "\n\n";

  std::cout << "Eigen solution: \n";
  for (unsigned i = 0; i < cols; i++) {
    std::cout << x_eigen_sparse[i] << std::endl;
  }
}
