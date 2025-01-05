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

// Define filename
#define CARP_CG_FILE "benchmark_results_carp_cg.csv"
#define EIGEN_CG_FILE "benchmark_results_eigen_cg.csv"
#define EIGEN_BICGSTAB_FILE "benchmark_results_eigen_bicgstab.csv"
#define CGMNC_FILE "benchmark_results_cgmnc.csv"
#define EIGEN_DIRECT_FILE "benchmark_results_eigen_direct.csv"
#define BASIC_KACZMARZ_FILE "benchmark_results_basic_kaczmarz.csv"
#define CUSOLVER_FILE "benchmark_results_cusolver.csv"

// Function declarations
void write_header(const std::string &file_path);
void compute_statistics(const std::vector<double> &times, double &avgTime,
                        double &stdDev);
unsigned int read_dimension(const std::string &file_path);
double calc_avgtime(const std::vector<double> &times);
int compute_bandwidth(const Eigen::SparseMatrix<double> &A);
BandedLinearSystem convert_to_banded(const SparseLinearSystem &sparse_system,
                                     unsigned bandwidth);

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

std::string generate_file_path(unsigned int problem, unsigned int complexity,
                               unsigned int degree);

std::string generate_file_path_banded(unsigned int problem,
                                      unsigned int complexity,
                                      unsigned int degree);

void add_elapsed_time_to_vec(
    std::vector<double> &times,
    const std::chrono::time_point<std::chrono::high_resolution_clock> start,
    const std::chrono::time_point<std::chrono::high_resolution_clock> end);

void write_result_to_file(const std::string &file_name, unsigned int problem,
                          unsigned int complexity, unsigned int degree,
                          double time, unsigned int dimension,
                          unsigned int num_it, unsigned int iteration,
                          KaczmarzSolverStatus status_input);

void inform_user_about_kaczmarz_status(KaczmarzSolverStatus status);

#endif  // BENCHMARKING_PDE_HPP
