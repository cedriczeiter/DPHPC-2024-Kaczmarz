#include "kaczmarz.hpp"
#include "kaczmarz_common.hpp"
#include "dense.hpp"
#include <cassert>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <random>

#define MAX_IT 1000000
double benchmark(int dim, int numIterations, double &stdDev, std::mt19937& rng) {
    std::vector<double> times;
    for (int i = 0; i < numIterations; ++i) {
        
        const DenseLinearSystem lse =  DenseLinearSystem::generate_random_regular(rng, dim);
    
        //Allocate memory to save kaczmarz solution
        std::vector<double> x_kaczmarz(dim, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        
       dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps selected arbitratly, we might need to revise this

        
        auto end = std::chrono::high_resolution_clock::now();
        
        
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
    std::mt19937 rng(21);
    
    // Open the file for output
    std::ofstream outFile("results.csv");
    outFile << "Dim,AvgTime,StdDev\n"; // Write the header for the CSV file
    
    // Loop over problem sizes, benchmark, and write to file
    for (int dim = 1; dim <= 32; dim *= 2) {
        double stdDev;
        double avgTime = benchmark(dim, numIterations, stdDev, rng);
        
        // Write results to the file
        outFile << dim << "," << avgTime << "," << stdDev << "\n";
    }
    
    outFile.close(); // Close the file after writing
    
    return 0;
}
