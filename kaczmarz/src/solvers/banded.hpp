#ifndef BANDED_HPP
#define BANDED_HPP

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "unpacked_banded_system.hpp"

class BandedSolver {
 public:
  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) = 0;

  virtual void setup(UnpackedBandedSystem* sys) = 0;

  virtual void flush_x() = 0;

  virtual void cleanup() = 0;

  virtual void iterate(unsigned iterations) = 0;

  void run_iterations(const BandedLinearSystem& lse, Vector& x,
                      unsigned iterations);

  KaczmarzSolverStatus solve(const BandedLinearSystem& lse, Vector& x,
                             unsigned iterations_step, unsigned max_iterations,
                             double abs_tolerance);
};

class CPUBandedSolver : public BandedSolver {
 protected:
  UnpackedBandedSystem* sys = nullptr;

 public:
  virtual void setup(UnpackedBandedSystem* sys);

  virtual void flush_x() {}

  virtual void cleanup();
};

class OpenMPGrouping1IBandedSolver : public CPUBandedSolver {
 private:
  const unsigned thread_count;

 public:
  OpenMPGrouping1IBandedSolver(const unsigned thread_count)
      : thread_count(thread_count) {}

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

  virtual void iterate(unsigned iterations);
};

class OpenMPGrouping2IBandedSolver : public CPUBandedSolver {
 private:
  const unsigned thread_count;

 public:
  OpenMPGrouping2IBandedSolver(const unsigned thread_count)
      : thread_count(thread_count) {}

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

  virtual void iterate(unsigned iterations);
};

class SerialNaiveBandedSolver : public CPUBandedSolver {
 public:
  SerialNaiveBandedSolver() {}

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

  virtual void iterate(unsigned iterations);
};

class SerialInterleavedBandedSolver : public CPUBandedSolver {
 public:
  SerialInterleavedBandedSolver() {}

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth);

  virtual void iterate(unsigned iterations);
};

#endif  // BANDED_HPP
