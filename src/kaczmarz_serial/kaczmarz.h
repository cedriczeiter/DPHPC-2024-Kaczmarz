#ifndef KACZMARZ_H
#define KACZMARZ_H

/**
 * Solves the Ax = b LSE by running the serial Kaczmarz algorithm with in-order row updates.
 *
 * The final iterate will be stored in the array x. The initial value of array x at the time of calling this function will be used as the initial guess.
 */
void kaczmarz_solver(const double *A, const double *b, double *x, unsigned rows, unsigned cols, unsigned max_iterations, double precision);

#endif // KACZMARZ_H
