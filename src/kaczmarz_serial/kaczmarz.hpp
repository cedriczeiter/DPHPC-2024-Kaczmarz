#ifndef KACZMARZ_HPP
#define KACZMARZ_HPP

enum class KaczmarzSolverStatus{
  Converged,
  ZeroNormRow,
  OutOfIterations
};

/**
 * Solves the Ax = b LSE by running the serial Kaczmarz algorithm with in-order row updates.
 *
 * The final iterate will be stored in the array x. The initial value of array x at the time of calling this function will be used as the initial guess.
 */
KaczmarzSolverStatus kaczmarz_solver(const double *A, const double *b, double *x, unsigned rows, unsigned cols, unsigned max_iterations, double precision);

#endif // KACZMARZ_HPP
