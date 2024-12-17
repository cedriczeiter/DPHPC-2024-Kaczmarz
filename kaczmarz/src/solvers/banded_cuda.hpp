#ifndef BANDED_CUDA_HPP
#define BANDED_CUDA_HPP

#include "unpacked_banded_system.hpp"

void invoke_kaczmarz_banded_cuda_grouping1(UnpackedBandedSystem& sys,
                                           const unsigned iterations,
                                           const unsigned block_count,
                                           const unsigned threads_per_block);

void invoke_kaczmarz_banded_cuda_grouping2(UnpackedBandedSystem& sys,
                                           const unsigned iterations,
                                           const unsigned block_count,
                                           const unsigned threads_per_block);

#endif  // BANDED_CUDA_HPP
