#ifndef RANDOM_DENSE_SYS_HPP
#define RANDOM_DENSE_SYS_HPP

#include <random>

/**
 * @brief Generates a random dense linear system Ax = b and solves for x wit LU-decomposition.
 *
 * This function generates a dense random matrix A and a vector b, then solves
 * the system of equations Ax = b for the unknown vector x. The matrix A is guaranteed
 * to be full-rank. The system is generated using a provided random number generator.
 *
 * @param rng A random number generator (std::mt19937) used for generating random values.
 * @param A A pointer to the matrix A (stored in row-major order) with dimensions dim x dim. Used to return system values.
 * @param b A pointer to the vector b, with dim elements. Used to return system values.
 * @param x A pointer to the vector x, where the solution to the system will be stored, with dim elements. Used to return system values.
 * @param dim The dimension of the system (number of rows/columns in A and elements in b and x).
 */
void generate_random_dense_linear_system(std::mt19937& rng, double* A, double* b, double* x, unsigned dim);

#endif // RANDOM_DENSE_SYS_HPP
