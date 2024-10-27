#ifndef KACZMARZ_TESTS_HPP
#define KACZMARZ_TESTS_HPP

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "kaczmarz.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"


/// @brief Runs tests on dense linear systems to compare Kaczmarz solution with Eigen's solution.
/// 
/// This function generates a random dense linear system of a given dimension,
/// solves it using the Kaczmarz algorithm, and compares the result to the solution 
/// obtained using Eigen's linear solver.
///
/// @param dim The dimension of the dense linear system (number of variables).
/// @param no_runs The number of test runs to perform for the given dimension.
void run_dense_tests(const unsigned dim, const unsigned no_runs);

/// @brief Runs tests on sparse linear systems to compare Kaczmarz solution with Eigen's solution.
/// 
/// This function generates a random sparse linear system of a given dimension and bandwidth,
/// solves it using the Kaczmarz algorithm, and compares the result to the solution 
/// obtained using Eigen's linear solver.
///
/// @param dim The dimension of the sparse linear system (number of variables).
/// @param bandwidth The bandwidth of the sparse linear system.
/// @param no_runs The number of test runs to perform for the given dimension and bandwidth.
void run_sparse_tests(const unsigned dim, const unsigned bandwidth, const unsigned no_runs);

#endif // KACZMARZ_TESTS_HPP
