#include "random_dense_system.hpp"
#include <iostream>
extern "C" { 
  #include "kaczmarz.h" 
} 
#include "gtest/gtest.h"
#define MAX_IT 200000
TEST(KaczmarzSerialDenseCorrectnessSmall, AgreesWithEigen){
  for (int i = 0; i < 30; i++){
    unsigned dim = 10;
    double *A = (double *)malloc(sizeof(double)*dim*dim);
    double *b = (double *)malloc(sizeof(double)*dim);
    double *x = (double *)malloc(sizeof(double)*dim);
    
    get_dense_linear_system(A, b, x, dim); //get randomised system

    double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);

    kaczmarz_solver(A, b, x_kaczmarz, dim, dim, MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps

    for (unsigned i = 0; i < dim; i++){
         ASSERT_LE(std::abs(x[i] - x_kaczmarz[i]), 1e-7);
    }
    free(A);
    free(b);
    free(x);
    free(x_kaczmarz);
  }
}


TEST(KaczmarzSerialDenseCorrectnessMedium, AgreesWithEigen){
  for (int i = 0; i < 30; i++){
    unsigned dim = 100;
    double *A = (double *)malloc(sizeof(double)*dim*dim);
    double *b = (double *)malloc(sizeof(double)*dim);
    double *x = (double *)malloc(sizeof(double)*dim);
    
    get_dense_linear_system(A, b, x, dim); //get randomised system

    double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);

    kaczmarz_solver(A, b, x_kaczmarz, dim, dim, MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps

    for (unsigned i = 0; i < dim; i++){
         ASSERT_LE(std::abs(x[i] - x_kaczmarz[i]), 1e-7);
    }
    free(A);
    free(b);
    free(x);
    free(x_kaczmarz);
  }
}
  
  
TEST(KaczmarzSerialDenseCorrectnessLarge, AgreesWithEigen){
  for (int i = 0; i < 30; i++){
    unsigned dim = 1000;
    double *A = (double *)malloc(sizeof(double)*dim*dim);
    double *b = (double *)malloc(sizeof(double)*dim);
    double *x = (double *)malloc(sizeof(double)*dim);
    
    get_dense_linear_system(A, b, x, dim); //get randomised system

    double *x_kaczmarz = (double *)malloc(sizeof(double)*dim);

    kaczmarz_solver(A, b, x_kaczmarz, dim, dim, MAX_IT*dim, 1e-10);//solve randomised system, max iterations steps

    for (unsigned i = 0; i < dim; i++){
         ASSERT_LE(std::abs(x[i] - x_kaczmarz[i]), 1e-7);
    }
    free(A);
    free(b);
    free(x);
    free(x_kaczmarz);
  }
}

int main() {
  std::srand(6);
  testing::InitGoogleTest();
  RUN_ALL_TESTS();
}
