#include "basic_cuda.hpp"

#include <chrono>
#include <cmath>
#include <iostream>


// Not implemented yet. This is just  a copy of the serial version.
KaczmarzSolverStatus dense_kaczmarz_cuda(const DenseLinearSystem &lse, double *x,
                                    const unsigned max_iterations,
                                    const double precision,
                                    std::vector<double> &times_residuals,
                                    std::vector<double> &residuals,
                                    std::vector<int> &iterations,
                                    const int convergence_step_rate) {
  const unsigned rows = lse.row_count();
  const unsigned cols = lse.column_count();

  std::cout << "dense_kaczmarz_cuda starting" << std::endl;

  // Calculate the residual norm in the beginning to check for convergence
  double residual_norm_sq = 0.0;
  for (unsigned k = 0; k < rows; k++) {
    double row_residual = 0.0;
    const double *row = lse.A() + k * cols;
    for (unsigned j = 0; j < cols; j++) {
      row_residual += row[j] * x[j];
    }
    row_residual -= lse.b()[k];
    residual_norm_sq += row_residual * row_residual;
  }

  const double residual_norm_0 = std::sqrt(residual_norm_sq);
  const auto start = std::chrono::high_resolution_clock::now();

  double residual_norm_now = 0;  // Preallocate to save allocation overhead

  // Iterate through a maximum of max_iterations
  for (unsigned iter = 0; iter < max_iterations; iter++) {
    if (iter % convergence_step_rate == 0) {
      const auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      times_residuals.push_back(elapsed.count());
      double residual_norm_sq = 0.0;
      
      
      for (unsigned k = 0; k < rows; k++) {
        double row_residual = 0.0;
        const double *row = lse.A() + k * cols;
        for (unsigned j = 0; j < cols; j++) {
          row_residual += row[j] * x[j];
        }
        row_residual -= lse.b()[k];
        residual_norm_sq += row_residual * row_residual;
      }

      residual_norm_now = std::sqrt(residual_norm_sq);
      residuals.push_back(residual_norm_now /
                          residual_norm_0);  // Takes residual fraction
      iterations.push_back(iter);

      // if residual converged enough, return
      if (residual_norm_now < precision) {
        return KaczmarzSolverStatus::Converged;
      }
    }



    ///////////////////////////////
    // From here on the relevant stuff to parallelize
    ///////////////////////////////

    // Process each row of matrix A
    
    double smallestRowSqNorm = updateAllRowsParallel(lse, x, rows, cols);
    if (smallestRowSqNorm < 1e-10) {
      return KaczmarzSolverStatus::ZeroNormRow;
    }
  }

  // If it didnt return earlier, then max iterations reached and not
  // converged.
  return KaczmarzSolverStatus::OutOfIterations;
}

