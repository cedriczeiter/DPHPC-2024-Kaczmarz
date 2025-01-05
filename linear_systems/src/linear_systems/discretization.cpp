#include "discretization.hpp"

Discretization Discretization::read_from_stream(std::istream &in_stream) {
  const SparseLinearSystem sys =
      SparseLinearSystem::read_from_stream(in_stream);
  const unsigned dim = sys.A().cols();
  std::vector<PositionHint> positions_hints(dim);
  for (unsigned i = 0; i < dim; i++) {
    in_stream >> positions_hints[i].x >> positions_hints[i].y;
  }
  return {sys, positions_hints};
}

void Discretization::write_to_stream(std::ostream &out_stream) const {
  this->sys.write_to_stream(out_stream);
  for (const auto &hint : this->position_hints) {
    out_stream << hint.x << ' ' << hint.y << '\n';
  }
}
