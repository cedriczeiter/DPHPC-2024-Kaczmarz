#include "random_dense_system.hpp"
#include "kaczmarz.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <cstring>
#include <random>

#define MAX_IT 1000000
#define RUNS_PER_DIM 5

void run_random_system_tests(const unsigned dim, const unsigned no_runs) {
  std::mt19937 rng(21);
  for (int i = 0; i < no_runs; i++){
    double *A = (double *)malloc(sizeof(double)*dim*dim);
    double *b = (double *)malloc(sizeof(double)*dim);
    double *x = (double *)malloc(sizeof(double)*dim);
    
    generate_random_dense_linear_system(rng, A, b, x, dim); //get randomised system including solution in x

    //Allocate memory to save kaczmarz solution
    double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);
    std::memset(x_kaczmarz, 0, sizeof(double) * dim); //set everything to zero in x_kaczmnarz

    kaczmarz_solver(A, b, x_kaczmarz, dim, dim, MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps
    //selected arbitratly, we might need to revise this

    for (unsigned j = 0; j < dim; j++){
         ASSERT_LE(std::abs(x[j] - x_kaczmarz[j]), 1e-6);
    }
    free(A);
    free(b);
    free(x);
    free(x_kaczmarz);
  }
}


TEST(KaczmarzSerialDenseCorrectnessSmall, AgreesWithEigen){
  run_random_system_tests(5, RUNS_PER_DIM);
}


TEST(KaczmarzSerialDenseCorrectnessMedium, AgreesWithEigen){
  run_random_system_tests(20, RUNS_PER_DIM);
}
  
  
TEST(KaczmarzSerialDenseCorrectnessLarge, AgreesWithEigen){
  run_random_system_tests(50, RUNS_PER_DIM);
}

int main() {
  testing::InitGoogleTest();
  return RUN_ALL_TESTS();
}
