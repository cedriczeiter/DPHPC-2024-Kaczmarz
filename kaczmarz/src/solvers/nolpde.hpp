#ifndef NOLPDE_HPP
#define NOLPDE_HPP

#include <memory>

#include "common.hpp"
#include "linear_systems/discretization.hpp"

class NolPDESolver {
 private:
  virtual void setup(const Discretization* d, Vector* x) = 0;

  virtual void flush_x() = 0;

  virtual void iterate(unsigned iterations) = 0;

  virtual void cleanup() = 0;

 public:
  void run_iterations(const Discretization& d, Vector& x, unsigned iterations);

  KaczmarzSolverStatus solve(const Discretization& lse, Vector& x,
                             unsigned iterations_step, unsigned max_iterations,
                             double abs_tolerance);
};

// NOTE: this permuting solver adapter/interface rounds groups to multiples of 2
// in both dimensions. That might be unnecessary for CPU-parallel
// implementations? Do we care?

class PermutingNolPDESolver : public NolPDESolver {
 protected:
  Vector post_permute_x;
  std::unique_ptr<SparseLinearSystem> post_permute_sys;

  // block count = length of `block_boundaries` - 1
  std::vector<unsigned> block_boundaries;

 private:
  Vector* x;
  std::vector<unsigned> permutation;
  std::vector<unsigned> inv_permutation;

  // if this returns a too large of a number or somehow can't be divided nicely,
  // some blocks might end up with no work
  virtual unsigned get_block_count_required() = 0;

  virtual void setup(const Discretization* d, Vector* x) override;

  virtual void flush_x() override;

  virtual void post_permute_setup() = 0;

  virtual void post_permute_flush_x() = 0;
};

class CUDANolPDESolver : public PermutingNolPDESolver {
 private:
  const unsigned block_count;
  const unsigned threads_per_block;

  double* x_gpu = nullptr;
  double* sq_norms_gpu = nullptr;
  double* b_gpu = nullptr;
  unsigned* A_outer_gpu = nullptr;
  unsigned* A_inner_gpu = nullptr;
  double* A_values_gpu = nullptr;
  unsigned* block_boundaries_gpu = nullptr;

  virtual unsigned get_block_count_required() override;

  virtual void post_permute_setup() override;

  virtual void post_permute_flush_x() override;

  virtual void iterate(unsigned iterations) override;

  virtual void cleanup() override;

 public:
  CUDANolPDESolver(const unsigned block_count, const unsigned threads_per_block)
      : block_count(block_count), threads_per_block(threads_per_block) {}
};

class BasicSerialNolPDESolver : public NolPDESolver {
 private:
  const Discretization* d = nullptr;
  Vector* x = nullptr;

  virtual void setup(const Discretization* d, Vector* x) override;

  virtual void flush_x() override;

  virtual void iterate(unsigned iterations) override;

  virtual void cleanup() override;
};

#endif  // NOLPDE_HPP
