#ifndef BANDED_HPP
#define BANDED_HPP

#include <vector>

#include "common.hpp"
#include "linear_systems/sparse.hpp"

/**
 * This struct is intended to hold the data of a banded LSE. Importantly, it
 * should do so in a way that is easy to work with in the solvers. A significant
 * difference to the storage in a `BandedLinearSystem` is that the full `2 *
 * bandwidth + 1` coefficients are explicitly for each row of the stiffness
 * matrix A; we store explicit zeros for entries that are 'outside the matrix'
 * for rows that are among the first or last `bandwidth` rows at the top or
 * bottom of the matrix.
 */
struct UnpackedBandedSystem {
  unsigned bandwidth;
  unsigned dim;
  std::vector<double> A_data;
  std::vector<double> x_padded;
  std::vector<double> sq_norms;
  std::vector<double> b;
};

/**
 * An abstract interface for a Kaczmarz solver solving a banded LSE.
 * The concrete implementation will need to provide implementations for the
 * virtual methods. This allows us to have unified implementations of the
 * non-virtual methods, which call the virtual ones.
 */
class BandedSolver {
 private:
  /**
   * The concrete solver implementation might work only for specific dimensions
   * of the LSE (something like: it must a multiple of the number of threads the
   * solver intends to run). In an implementation of this method, the solver
   * should return the new ('padded') dimension of the LSE that has the desired
   * property and that is greater equal to the provided `dim`.
   */
  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) = 0;

  /**
   * Sets what system the solver should operate on. The provided `sys` is
   * guaranteed to have the dimension that has been returned by `pad_dimension`.
   */
  virtual void setup(UnpackedBandedSystem* sys) = 0;

  /**
   * Makes sure that the current iterate for `x` is written to the appropriate
   * of the `sys` that has been provided in the last `setup` call.
   */
  virtual void flush_x() = 0;

  /**
   * Deinitialized resources initialized by `setup`.
   */
  virtual void cleanup() = 0;

  /**
   * Run a fixed number of iterations on the LSE given in the last call to
   * `setup`. Does not necessarily update the `x` iterate; see the `flush_x`
   * method.
   */
  virtual void iterate(unsigned iterations) = 0;

 public:
  /**
   * Runs a fixed number of Kaczmarz iterations (Kaczmarz sweeps)
   */
  void run_iterations(const BandedLinearSystem& lse, Vector& x,
                      unsigned iterations);

  /**
   * Runs Kaczmarz iterations until the L2 norm of the residual is below
   * `abs_tolerance`; but also stops if `max_iterations` are performed before
   * that happens. The residual is only checked in between batches of
   * iterations; each batch performs `iterations_step` iterations.
   */
  KaczmarzSolverStatus solve(const BandedLinearSystem& lse, Vector& x,
                             unsigned iterations_step, unsigned max_iterations,
                             double abs_tolerance);
};

/**
 * A common subclass for solvers that don't use the GPU.
 * They don't need to allocate any new memory and they directly modify the `x`
 * iterate in the provided `sys`, so their `setup`, `flush_x`, and `cleanup`
 * methods are simple.
 */
class CPUBandedSolver : public BandedSolver {
 protected:
  UnpackedBandedSystem* sys = nullptr;

 private:
  virtual void setup(UnpackedBandedSystem* sys) override;

  virtual void flush_x() override {}

  virtual void cleanup() override;
};

/**
 * The "Grouping1" refers to how the rows are scheduled/grouped together for
 * parallel processing.
 */
class OpenMPGrouping1BandedSolver : public CPUBandedSolver {
 private:
  const unsigned thread_count;

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) override;

  virtual void iterate(unsigned iterations) override;

 public:
  OpenMPGrouping1BandedSolver(const unsigned thread_count)
      : thread_count(thread_count) {}
};

/**
 * The "Grouping2" refers to how the rows are scheduled/grouped together for
 * parallel processing.
 */
class OpenMPGrouping2BandedSolver : public CPUBandedSolver {
 private:
  const unsigned thread_count;

  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) override;

  virtual void iterate(unsigned iterations) override;

 public:
  OpenMPGrouping2BandedSolver(const unsigned thread_count)
      : thread_count(thread_count) {}
};

/**
 * The "Naive" refers to the order in which the rows are processed in each
 * sweep. In this case, it is just all rows top to bottom.
 */
class SerialNaiveBandedSolver : public CPUBandedSolver {
 private:
  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) override;

  virtual void iterate(unsigned iterations) override;

 public:
  SerialNaiveBandedSolver() {}
};

/**
 * The "Interleaved" refers to the order in which the rows are processed in each
 * sweep. In this case, all `row_idx mod (2 * bandwidth + 1) = 0` are processed
 * top to  bottom; then all `... mod ... = 1`; then all `... mod ... = 2`; etc.
 * We do this interleaving because empirically, it seems to have better
 * performance than the naive order.
 */
class SerialInterleavedBandedSolver : public CPUBandedSolver {
 private:
  virtual unsigned pad_dimension(unsigned dim, unsigned bandwidth) override;

  virtual void iterate(unsigned iterations) override;

 public:
  SerialInterleavedBandedSolver() {}
};

#endif  // BANDED_HPP
