#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>

/**
 * @brief Enumeration to describe the result of the Kaczmarz solver.
 */
enum class KaczmarzSolverStatus { Converged, ZeroNormRow, OutOfIterations };

std::string kaczmarz_status_string(const KaczmarzSolverStatus status);

#endif  // COMMON_HPP
