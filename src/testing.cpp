#include "kaczmarz.hpp"

#include "linear_systems/dense.hpp"

#include "gtest/gtest.h"
#include <cmath>
#include <random>

constexpr unsigned MAX_IT = 1000000;
constexpr unsigned RUNS_PER_DIM = 5;

void run_random_system_tests(const unsigned dim, const unsigned no_runs) {
  std::mt19937 rng(21);
  for (unsigned i = 0; i < no_runs; i++){

    const DenseLinearSystem lse = DenseLinearSystem::generate_random_regular(rng, dim);
    
    // Allocate memory to save kaczmarz solution
    // Set everything to zero in x_kaczmnarz
    std::vector<double> x_kaczmarz(dim, 0.0);

    // precision and max. iterations selected randomly, we might need to revise this
    dense_kaczmarz(lse, &x_kaczmarz[0], MAX_IT*dim, 1e-10);

    const Vector x_eigen = lse.eigen_solve();

    for (unsigned i = 0; i < dim; i++){
         ASSERT_LE(std::abs(x_eigen[i] - x_kaczmarz[i]), 1e-6);
    }
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
