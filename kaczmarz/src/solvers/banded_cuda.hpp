#ifndef BANDED_CUDA_HPP
#define BANDED_CUDA_HPP

#include "unpacked_banded_system.hpp"
#include "banded.hpp"

class GPUBandedSolver : public BandedSolver {
  protected:
    UnpackedBandedSystem *sys = nullptr;
    double *x_gpu = nullptr;
    double *A_data_gpu = nullptr;
    double *sq_norms_gpu = nullptr;
    double *b_gpu = nullptr;

  public:
    virtual void setup(UnpackedBandedSystem* sys);

    virtual void flush_x();

    virtual void cleanup();
};

class CUDAGrouping1BandedSolver : public GPUBandedSolver {
  private:
    const unsigned block_count;
    const unsigned threads_per_block;

  public:
    CUDAGrouping1BandedSolver(const unsigned block_count, const unsigned threads_per_block) : block_count(block_count), threads_per_block(threads_per_block) { }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void iterate(unsigned iterations);
};

class CUDAGrouping2BandedSolver : public GPUBandedSolver {
  private:
    const unsigned block_count;
    const unsigned threads_per_block;

  public:
    CUDAGrouping2BandedSolver(const unsigned block_count, const unsigned threads_per_block) : block_count(block_count), threads_per_block(threads_per_block) { }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void iterate(unsigned iterations);
};

#endif  // BANDED_CUDA_HPP
