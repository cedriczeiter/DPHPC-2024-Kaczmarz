#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "kaczmarz.hpp"
#include "kaczmarz_common.hpp"
#include "linear_systems/dense.hpp"

/// @file benchmark.hpp
/// @brief Contains the benchmark function declaration for measuring Kaczmarz
/// algorithm performance.

/// @brief Benchmarks the Kaczmarz algorithm on a randomly generated dense
/// linear system.
///
/// This function generates a random dense linear system of a given dimension
/// and solves it using the Kaczmarz algorithm multiple times to gather
/// statistics on average runtime and standard deviation.
///
/// @param dim The dimension of the linear system (number of variables).
/// @param numIterations Number of iterations for averaging results.
/// @param stdDev Reference to a variable where the computed standard deviation
/// will be stored.
/// @param rng A random number generator used for creating the linear system.
/// @return The average time taken to solve the system across all iterations.
double benchmark(int dim, int numIterations, double& stdDev, std::mt19937& rng);

#endif  // BENCHMARK_HPP
