#ifndef COMMON_HPP
#define COMMON_HPP

#include <cassert>
#include <string>

/**
 * @brief Enumeration to describe the result of the Kaczmarz solver.
 */
enum class KaczmarzSolverStatus { Converged, ZeroNormRow, OutOfIterations };

std::string kaczmarz_status_string(const KaczmarzSolverStatus status);

inline unsigned ceil_div(const unsigned a, const unsigned b) {
  assert(b != 0);
  if (a == 0) {
    return 0;
  }
  return (a - 1) / b + 1;
}

#endif  // COMMON_HPP
