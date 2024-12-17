#ifndef UNPACKED_BANDED_SYSTEM_HPP
#define UNPACKED_BANDED_SYSTEM_HPP

#include <vector>

struct UnpackedBandedSystem {
  unsigned bandwidth;
  unsigned dim;
  std::vector<double> A_data;
  std::vector<double> x_padded;
  std::vector<double> sq_norms;
  std::vector<double> b;
};

#endif  // UNPACKED_BANDED_SYSTEM_HPP
