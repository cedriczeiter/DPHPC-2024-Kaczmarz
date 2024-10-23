#include "random_dense_system.hpp"
#include "kaczmarz.hpp"
#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
//#include "gtest/gtest.h"
#include <cmath>
#include <cstring>
#include <random>

#define MAX_IT 1000000
double benchmark(int dim, int numIterations, double &stdDev) {
    std::vector<double> times;
    for (int i = 0; i < numIterations; ++i) {
        
        std::mt19937 rng(21);
        double *A = (double *)malloc(sizeof(double)*dim*dim);
        double *b = (double *)malloc(sizeof(double)*dim);
        double *x = (double *)malloc(sizeof(double)*dim);
        
        generate_random_dense_linear_system(rng, A, b, x, dim); //get randomised system including solution in x

        //Allocate memory to save kaczmarz solution
        double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);
        std::memset(x_kaczmarz, 0, sizeof(double) * dim); //set everything to zero in x_kaczmnarz

        
        
        auto start = std::chrono::high_resolution_clock::now();
        
        
        // Run the heat diffusion simulation
        kaczmarz_solver(A, b, x_kaczmarz, dim, dim, MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps
        //selected arbitratly, we might need to revise this

        
        auto end = std::chrono::high_resolution_clock::now();
        
        free(A);
        free(b);
        free(x);
        free(x_kaczmarz);
        
        std::chrono::duration<double> elapsed = end - start;
        
        times.push_back(elapsed.count());
    }
    
    // Calculate average time
    double avgTime = 0;
    for (double time : times) {
        avgTime += time;
    }
    avgTime /= numIterations;
    
    // Calculate standard deviation
    double variance = 0;
    for (double time : times) {
        variance += (time - avgTime) * (time - avgTime);
    }
    variance /= numIterations;
    stdDev = std::sqrt(variance);
    
    return avgTime;
}

int main() {
    int numIterations = 10; // Number of iterations to reduce noise
    
    // Open the file for output
    std::ofstream outFile("results.csv");
    outFile << "Dim,AvgTime,StdDev\n"; // Write the header for the CSV file
    
    // Loop over problem sizes, benchmark, and write to file
    for (int dim = 1; dim <= 32; dim *= 2) {
        double stdDev;
        double avgTime = benchmark(dim, numIterations, stdDev);
        
        // Write results to the file
        outFile << dim << "," << avgTime << "," << stdDev << "\n";
    }
    
    outFile.close(); // Close the file after writing
    
    return 0;
}
