#ifndef BANDED_CUDA_HPP
#define BANDED_CUDA_HPP

#include <vector>

void invoke_kaczmarz_banded_update(const unsigned bandwidth,
                                   const unsigned thread_count,
                                   const std::vector<double>& A_data_padded,
                                   std::vector<double>& x_padded,
                                   const std::vector<double>& sq_norms_padded,
                                   const std::vector<double>& b_padded);

#endif  // BANDED_CUDA_HPP
