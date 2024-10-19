#ifndef RANDOM_DENSE_SYS_HPP
#define RANDOM_DENSE_SYS_HPP


/**
 * To get deterministic results, seed with std::srand before calling this function.
 */
void generate_random_dense_linear_system(double* A, double* b, double* x, unsigned dim);


#endif // RANDOM_DENSE_SYS_HPP
