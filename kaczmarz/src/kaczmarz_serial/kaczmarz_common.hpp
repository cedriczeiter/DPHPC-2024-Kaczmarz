#ifndef KACZMARZ_COMMON_HPP
#define KACZMARZ_COMMON_HPP

#include <cassert>
#include <string>

/**
 * @brief Enumeration to describe the result of the Kaczmarz solver.
 */
enum class KaczmarzSolverStatus { Converged, ZeroNormRow, OutOfIterations };

std::string kaczmarz_status_string(const KaczmarzSolverStatus status);

#endif  // KACZMARZ_COMMON_HPP
