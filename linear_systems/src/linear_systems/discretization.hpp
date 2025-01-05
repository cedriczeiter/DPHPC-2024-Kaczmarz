#ifndef DISCRETIZATION_HPP
#define DISCRETIZATION_HPP

#include <vector>

#include "sparse.hpp"

struct PositionHint {
  double x, y;
};

struct Discretization {
  SparseLinearSystem sys;
  std::vector<PositionHint> position_hints;

  static Discretization read_from_stream(std::istream &in_stream);

  void write_to_stream(std::ostream &out_stream) const;
};

#endif  // DISCRETIZATION_HPP
