#include "common.hpp"

#include <cassert>

std::string kaczmarz_status_string(const KaczmarzSolverStatus status) {
  switch (status) {
    case KaczmarzSolverStatus::Converged:
      return "converged";
    case KaczmarzSolverStatus::ZeroNormRow:
      return "zero-norm row";
    case KaczmarzSolverStatus::OutOfIterations:
      return "out of iterations";
  }
  assert(!"invalid Kaczmarz solver status!");
  return "";
}
