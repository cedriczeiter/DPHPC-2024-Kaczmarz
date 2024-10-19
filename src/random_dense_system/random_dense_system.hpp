#ifndef RANDOM_DENSE_SYS_HPP
#define RANDOM_DENSE_SYS_HPP


#include <random>

void generate_random_dense_linear_system(std::mt19937& rng, double* A, double* b, double* x, unsigned dim);


#endif // RANDOM_DENSE_SYS_HPP
