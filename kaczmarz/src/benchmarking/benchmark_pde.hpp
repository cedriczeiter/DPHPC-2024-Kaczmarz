#ifndef BENCHMARKING_PDE_HPP
#define BENCHMARKING_PDE_HPP

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include "asynchronous.hpp"
#include "banded.hpp"
#include "basic.hpp"
#include "carp.hpp"
#include "cusolver.hpp"
#include "linear_systems/dense.hpp"
#include "linear_systems/sparse.hpp"
#include "random.hpp"
#include "sparse_cg.hpp"

// Define constants

// Maximum number of iterations
#define MAX_IT (std::numeric_limits<unsigned int>::max() - 1)
// Precision threshold
#define PRECISION 1e-9
// Highest complexity to test
#define MAX_COMPLEXITY 8
// Highest degree to test (out of three currently available)
#define MAX_DEGREE 1
// Number of iterations for each benchmark
#define NUM_IT 10
// Cutoff time for benchmarking
#define TIME_THRESHOLD 300.0

// make sure to change L_RESIDUAL in carp_utils.hpp to 1000

// Function declarations
void write_header(const std::string &file_path);
int compute_bandwidth(const Eigen::SparseMatrix<double> &A);
BandedLinearSystem convert_to_banded(const SparseLinearSystem &sparse_system,
                                     unsigned bandwidth);
void compute_statistics(const std::vector<double> &times, double &avgTime,
                        double &stdDev);
void write_results_to_file(const std::string &file_name, unsigned int problem,
                           unsigned int complexity, unsigned int degree,
                           double avg_time, double std_dev,
                           unsigned int dimension);

double benchmark_carpcg(unsigned int numIterations, unsigned int problem_i,
                        unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_cg(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_bicgstab(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i);
double benchmark_cgmnc(unsigned int numIterations, unsigned int problem_i,
                       unsigned int complexity_i, unsigned int degree_i);
double benchmark_eigen_direct(unsigned int numIterations,
                              unsigned int problem_i, unsigned int complexity_i,
                              unsigned int degree_i);
double benchmark_basic_kaczmarz(unsigned int numIterations,
                                unsigned int problem_i,
                                unsigned int complexity_i,
                                unsigned int degree_i);
double benchmark_cusolver(unsigned int numIterations, unsigned int problem_i,
                          unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_cuda(unsigned int numIterations, unsigned int problem_i,
                             unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_cpu(unsigned int numIterations, unsigned int problem_i,
                            unsigned int complexity_i, unsigned int degree_i);
double benchmark_banded_serial(unsigned int numIterations,
                               unsigned int problem_i,
                               unsigned int complexity_i,
                               unsigned int degree_i);

SparseLinearSystem read_matrix_from_file(const std::string &file_path);

void write_results(const std::string &file_name, unsigned int problem,
                   unsigned int complexity, unsigned int degree,
                   double avg_time, double std_dev,
                   const std::string &file_path);

std::string generate_file_path(unsigned int problem, unsigned int complexity,
                               unsigned int degree);

void add_elapsed_time_to_vec(
    std::vector<double> &times,
    const std::chrono::time_point<std::chrono::high_resolution_clock> start,
    const std::chrono::time_point<std::chrono::high_resolution_clock> end);

double write_and_calc_results(const std::string &file_name,
                              unsigned int problem, unsigned int complexity,
                              unsigned int degree, const std::string &file_path,
                              const std::vector<double> &times);

void inform_user_about_kaczmarz_status(KaczmarzSolverStatus status);

#endif  // BENCHMARKING_PDE_HPP