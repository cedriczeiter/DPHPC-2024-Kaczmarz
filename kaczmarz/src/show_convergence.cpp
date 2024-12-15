#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>

#ifndef L_RESIDUAL
    #define L_RESIDUAL 1
#endif

#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"
#include "solvers/asynchronous.hpp"
#include "solvers/banded.hpp"
#include "solvers/carp.hpp"
#include "solvers/cusolver.hpp"
#include "solvers/basic.hpp"

using hrclock = std::chrono::high_resolution_clock;


#define NRUNS 10           // Number of precision steps
#define NUM_REPETITIONS 4  // Number of repetitions per precision

// Function to compute the standard deviation
double compute_stddev(const std::vector<double>& times, double mean) {
    double sum = 0.0;
    for (double t : times) {
        sum += (t - mean) * (t - mean);
    }
    return std::sqrt(sum / times.size());
}

int main() {
    // Open a single CSV file to write all results
    std::ofstream outFile("convergence_results.csv");
    if (!outFile) {
        std::cerr << "Error: Could not open results file for writing." << std::endl;
        return -1;
    }

    // Write the header for the CSV file
    outFile << "Problem,Complexity,Degree,Precision,AvgCarpTime,StdDevCarpTime,AvgNormalTime,StdDevNormalTime\n";

    // Iterate through all problems, complexities, and degrees
    for (int problem = 1; problem <= 3; ++problem) {
        for (int complexity = 1; complexity <= 3; ++complexity) {
            for (int degree = 1; degree <= 1; ++degree) {
                // Construct the file path dynamically
                std::ostringstream file_path;
                file_path << "../../generated_bvp_matrices/problem" << problem << "/problem" << problem
                          << "_complexity" << complexity << "_degree" << degree
                          << ".txt";

                std::cout << "Processing file: " << file_path.str() << std::endl;

                // Read in the system from the generated file
                std::ifstream lse_input_stream(file_path.str());
                if (!lse_input_stream) {
                    std::cerr << "Error: Could not open file " << file_path.str()
                              << std::endl;
                    continue; // Skip this file and move to the next one
                }

                const SparseLinearSystem sparse_lse =
                    SparseLinearSystem::read_from_stream(lse_input_stream);

                // Define Variables
                const unsigned dim = sparse_lse.row_count();
                const unsigned max_iterations = std::numeric_limits<unsigned int>::max() - 1;

                std::cout << "Dimension: " << dim << std::endl;

                double precision = 1; // Initial precision
                for (int precision_level = 0; precision_level < NRUNS; ++precision_level) {
                    //////////////////////////////////////////
                    // Repeated Runs to Calculate Avg and StdDev
                    //////////////////////////////////////////
                    std::vector<double> carp_times;
                    std::vector<double> normal_times;

                    for (int repetition = 0; repetition < NUM_REPETITIONS; ++repetition) {
                        //////////////////////////////////////////
                        // Calculating the solution with CARP
                        //////////////////////////////////////////
                        Vector x_kaczmarz = Vector::Zero(dim);

                        const auto kaczmarz_start = hrclock::now();
                        int nr_of_steps = 0;
                        const double relaxation = 0.35;
                        const auto status = carp_gpu(sparse_lse, x_kaczmarz, max_iterations,
                                                     precision, relaxation, nr_of_steps);
                        const auto kaczmarz_end = hrclock::now();
                        const auto kaczmarz_time =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                kaczmarz_end - kaczmarz_start)
                                .count();

                        carp_times.push_back(kaczmarz_time);

                        //////////////////////////////////////////
                        // Calculating the solution with Sparse Kaczmarz
                        //////////////////////////////////////////
                        Vector x_iter = Vector::Zero(dim);
                        std::vector<double> times_residuals;
                        std::vector<double> residuals;
                        std::vector<int> iterations;
                        const auto iter_start = hrclock::now();
                        sparse_kaczmarz(sparse_lse, x_iter, max_iterations, precision,
                                        times_residuals, residuals, iterations, 1);
                        const auto iter_end = hrclock::now();
                        const auto normal_time =
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                iter_end - iter_start)
                                .count();

                        normal_times.push_back(normal_time);
                    }

                    //////////////////////////////////////////
                    // Compute Average and Standard Deviation
                    //////////////////////////////////////////
                    double avg_carp_time = std::accumulate(carp_times.begin(), carp_times.end(), 0.0) / NUM_REPETITIONS;
                    double stddev_carp_time = compute_stddev(carp_times, avg_carp_time);

                    double avg_normal_time = std::accumulate(normal_times.begin(), normal_times.end(), 0.0) / NUM_REPETITIONS;
                    double stddev_normal_time = compute_stddev(normal_times, avg_normal_time);

                    //////////////////////////////////////////
                    // Write Results to CSV
                    //////////////////////////////////////////
                    outFile << problem << "," << complexity << "," << degree << ","
                            << precision << "," << avg_carp_time << "," << stddev_carp_time << ","
                            << avg_normal_time << "," << stddev_normal_time << "\n";

                    std::cout << "Precision: " << precision 
                              << " - Avg Carp Time: " << avg_carp_time 
                              << " - StdDev Carp Time: " << stddev_carp_time
                              << " - Avg Normal Time: " << avg_normal_time
                              << " - StdDev Normal Time: " << stddev_normal_time
                              << std::endl;

                    precision *= 0.1; // Decrease precision for the next level
                }

                std::cout << "Finished processing file: " << file_path.str() << std::endl;
            }
        }
    }

    std::cout << "All problems processed successfully!" << std::endl;

    return 0;
}