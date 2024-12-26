#ifndef BANDED_HPP
#define BANDED_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "unpacked_banded_system.hpp"

class BandedSolver {
  public:
    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) = 0;

    virtual void setup(UnpackedBandedSystem* sys) = 0;

    virtual void cleanup() = 0;

    virtual void iterate(unsigned iterations) = 0;
    void run_iterations(const BandedLinearSystem& lse, Vector& x, const unsigned iterations);
};

class OpenMPGrouping1IBandedSolver : public BandedSolver {
  private:
    UnpackedBandedSystem* sys = nullptr;
    const unsigned thread_count;

  public:
    OpenMPGrouping1IBandedSolver(const unsigned thread_count) : thread_count(thread_count) {
    }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void setup(UnpackedBandedSystem* sys);

    virtual void cleanup();

    virtual void iterate(unsigned iterations);
};

class OpenMPGrouping2IBandedSolver : public BandedSolver {
  private:
    UnpackedBandedSystem* sys = nullptr;
    const unsigned thread_count;

  public:
    OpenMPGrouping2IBandedSolver(const unsigned thread_count) : thread_count(thread_count) {
    }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void setup(UnpackedBandedSystem* sys);

    virtual void cleanup();

    virtual void iterate(unsigned iterations);
};

class SerialNaiveBandedSolver : public BandedSolver {
  private:
    UnpackedBandedSystem* sys = nullptr;

  public:
    SerialNaiveBandedSolver() {
    }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void setup(UnpackedBandedSystem* sys);

    virtual void cleanup();

    virtual void iterate(unsigned iterations);
};

class SerialInterleavedBandedSolver : public BandedSolver {
  private:
    UnpackedBandedSystem* sys = nullptr;

  public:
    SerialInterleavedBandedSolver() {
    }

    virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

    virtual void setup(UnpackedBandedSystem* sys);

    virtual void cleanup();

    virtual void iterate(unsigned iterations);
};

void kaczmarz_banded_cuda_grouping1(const BandedLinearSystem& lse, Vector& x,
                                    unsigned iterations, unsigned block_count,
                                    unsigned threads_per_block);

void kaczmarz_banded_cuda_grouping2(const BandedLinearSystem& lse, Vector& x,
                                    unsigned iterations, unsigned block_count,
                                    unsigned threads_per_block);

#endif  // BANDED_HPP
