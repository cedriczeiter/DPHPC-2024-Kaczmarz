#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>

#include "kaczmarz.hpp"
#include "kaczmarz_serial/kaczmarz_common.hpp"
#include "kaczmarz_serial/random_kaczmarz.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "random_kaczmarz.hpp"

int main() {
  const unsigned dim = 5;

  // Initialize the random number generator

  std::mt19937 rng(21);

  const DenseLinearSystem lse =
      DenseLinearSystem::generate_random_regular(rng, dim);

  std::vector<double> x_kaczmarz(dim, 0.0);
  std::vector<double> x_kaczmarz_random(dim, 0.0);

  const auto status_dense = dense_kaczmarz(lse, &x_kaczmarz[0], 100000, 1e-10);
  if (status_dense != KaczmarzSolverStatus::Converged) {
    std::cout << "The serial Kaczmarz solver didn't converge!" << std::endl;
  }

  const auto status_random =
      kaczmarz_random_solver(lse, &x_kaczmarz_random[0], 100000, 1e-10);
  if (status_random != KaczmarzSolverStatus::Converged) {
    std::cout << "The random Kaczmarz solver didn't converge!" << std::endl;
  }

  std::cout << "Kaczmarz solution: \n";
  for (unsigned i = 0; i < dim; i++) {
    std::cout << x_kaczmarz[i] << std::endl;
  }
  std::cout << "\n\n";

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

  // Set complete file path to the sparse sytem file
  const std::filesystem::path here = __FILE__;
  auto system_file_path =
      here.parent_path() / "../../system_matrices/elementmatrix_unitsquare.txt";
  // read in from file
  const SparseLinearSystem sparse_lse =
      SparseLinearSystem::read_from_file(system_file_path.string());
  // solve
  Eigen::VectorXd x_kaczmarz_sparse =
      Eigen::VectorXd::Zero(sparse_lse.column_count());
  const auto status_sparse =
      sparse_kaczmarz(sparse_lse, x_kaczmarz_sparse, 10000000, 1e-10);
  if (status_sparse != KaczmarzSolverStatus::Converged) {
    std::cout
        << "The serial Kaczmarz solver for sparse matrices didn't converge!"
        << std::endl;
  }

  unsigned cols = sparse_lse.column_count();

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
