#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "solvers/asynchronous.hpp"
#include "solvers/basic.hpp"

constexpr unsigned MAX_IT = 1000000;
constexpr unsigned RUNS_PER_DIM = 5;

void run_parallel_tests(const unsigned dim, const unsigned bandwidth,
                        const unsigned no_runs) {
  std::mt19937 rng(21);
  for (unsigned i = 0; i < no_runs; i++) {
    const SparseLinearSystem lse =
        SparseLinearSystem::generate_random_banded_regular(rng, dim, bandwidth);

    Vector x_kaczmarz = Vector::Zero(dim);

    auto result =
        sparse_kaczmarz_parallel(lse, x_kaczmarz, MAX_IT * dim, 1e-9, std::min(dim, 8u));

    ASSERT_EQ(KaczmarzSolverStatus::Converged, result);

    const Vector x_eigen = lse.eigen_solve();

    ASSERT_LE((x_kaczmarz - x_eigen).norm(), 1e-6);
  }
}

TEST(KaczmarzParallelSparseCorrectnessSmall, AgreesWithEigen) {
  run_parallel_tests(5, 1, RUNS_PER_DIM);
}

TEST(KaczmarzParallelSparseCorrectnessMedium, AgreesWithEigen) {
  run_parallel_tests(20, 2, RUNS_PER_DIM);
}

TEST(KaczmarzParallelSparseCorrectnessLarge, AgreesWithEigen) {
  run_parallel_tests(50, 2, RUNS_PER_DIM);
}

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
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;
    // precision and max. iterations selected randomly, we might need to revise
    // this
    dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT * dim, 1e-9, times_residuals,
                   residuals, iterations, MAX_IT);

    const Vector x_eigen = lse.eigen_solve();

    double norm = 0;
    for (int i = 0; i < dim; i++) {
      norm += (x_kaczmarz[i] - x_eigen[i]) * (x_kaczmarz[i] - x_eigen[i]);
    }
    norm = std::sqrt(norm);
    ASSERT_LE(norm, 1e-6);
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

    Vector x_kaczmarz = Vector::Zero(dim);

    // precision and max. iterations selected randomly, we might need to revise
    // this
    std::vector<double> times_residuals;
    std::vector<double> residuals;
    std::vector<int> iterations;

    auto result =
        sparse_kaczmarz(lse, x_kaczmarz, MAX_IT * dim, 1e-9, times_residuals,
                        residuals, iterations, MAX_IT);

    ASSERT_EQ(result, KaczmarzSolverStatus::Converged);

    const Vector x_eigen = lse.eigen_solve();

    ASSERT_LE((x_kaczmarz - x_eigen).norm(), 1e-6);
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
