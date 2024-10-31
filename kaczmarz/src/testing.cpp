#include <cmath>
#include <random>
#include <vector>

#include "kaczmarz.hpp"
#include "kaczmarz_asynchronous.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "gtest/gtest.h"

constexpr unsigned MAX_IT = 1000000;
constexpr unsigned RUNS_PER_DIM = 5;

/// @brief Runs tests on dense linear systems to compare Kaczmarz solution with
/// Eigen's solution.
///
/// This function generates a random dense linear system of a given dimension,
/// solves it using the Kaczmarz algorithm, and compares the result to the
/// solution obtained using Eigen's linear solver.
///
/// @param dim The dimension of the dense linear system (number of variables).
/// @param no_runs The number of test runs to perform for the given dimension.
void run_dense_tests(const unsigned dim, const unsigned no_runs) {
  std::mt19937 rng(21);
  for (unsigned i = 0; i < no_runs; i++) {
    const DenseLinearSystem lse =
        DenseLinearSystem::generate_random_regular(rng, dim);

    // Allocate memory to save kaczmarz solution
    // Set everything to zero in x_kaczmnarz
    std::vector<double> x_kaczmarz(dim, 0.0);

    // precision and max. iterations selected randomly, we might need to revise
    // this
    dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT * dim, 1e-10);

    const Vector x_eigen = lse.eigen_solve();

    for (unsigned i = 0; i < dim; i++) {
      ASSERT_LE(std::abs(x_eigen[i] - x_kaczmarz[i]), 1e-6);
    }
  }
}

TEST(KaczmarzSerialDenseCorrectnessSmall, AgreesWithEigen) {
  run_dense_tests(5, RUNS_PER_DIM);
}

TEST(KaczmarzSerialDenseCorrectnessMedium, AgreesWithEigen) {
  run_dense_tests(20, RUNS_PER_DIM);
}

TEST(KaczmarzSerialDenseCorrectnessLarge, AgreesWithEigen) {
  run_dense_tests(50, RUNS_PER_DIM);
}

/// @brief Runs tests on sparse linear systems to compare Kaczmarz solution with
/// Eigen's solution.
///
/// This function generates a random sparse linear system of a given dimension
/// and bandwidth, solves it using the Kaczmarz algorithm, and compares the
/// result to the solution obtained using Eigen's linear solver.
///
/// @param dim The dimension of the sparse linear system (number of variables).
/// @param bandwidth The bandwidth of the sparse linear system.
/// @param no_runs The number of test runs to perform for the given dimension
/// and bandwidth.
void run_sparse_tests(const unsigned dim, const unsigned bandwidth,
                      const unsigned no_runs) {
  std::mt19937 rng(21);
  for (unsigned i = 0; i < no_runs; i++) {
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, bandwidth);

    // Vector x_kaczmarz = Vector::Zero(dim);
    std::vector<double> x_kaczmarz(dim, 0.0);

    // precision and max. iterations selected randomly, we might need to revise
    // this
    sparse_kaczmarz_parallel(lse, &x_kaczmarz[0], MAX_IT * dim, 1e-10, 4);

    const Vector x_eigen = lse.eigen_solve();

    for (unsigned i = 0; i < dim; i++) {
      ASSERT_LE(std::abs(x_eigen[i] - x_kaczmarz[i]), 1e-6);
    }
  }
}

TEST(KaczmarzSerialSparseCorrectnessSmall, AgreesWithEigen) {
  run_sparse_tests(5, 1, RUNS_PER_DIM);
}

TEST(KaczmarzSerialSparseCorrectnessMedium, AgreesWithEigen) {
  run_sparse_tests(20, 2, RUNS_PER_DIM);
}

TEST(KaczmarzSerialSparseCorrectnessLarge, AgreesWithEigen) {
  run_sparse_tests(50, 2, RUNS_PER_DIM);
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
