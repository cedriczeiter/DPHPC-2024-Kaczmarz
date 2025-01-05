#include "nolpde.hpp"

void NolPDESolver::run_iterations(const Discretization& /* d */,
                                  Vector& /* x */, unsigned /* iterations */) {
  throw "TODO: implement";
}

KaczmarzSolverStatus NolPDESolver::solve(const Discretization& /* lse */,
                                         Vector& /* x */,
                                         unsigned /* iterations_step */,
                                         unsigned /* max_iterations */,
                                         double /* abs_tolerance */) {
  throw "TODO: implement";
}

void PermutingNolPDESolver::setup(const Discretization* const /* d */,
                                  Vector* const /* x */) {
  throw "TODO: implement";
}

void PermutingNolPDESolver::flush_x() { throw "TODO: implement"; }

unsigned CUDANolPDESolver::get_blocks_required() { throw "TODO: implement"; }

void CUDANolPDESolver::post_permute_setup() { throw "TODO: implement"; }

void CUDANolPDESolver::post_permute_flush_x() { throw "TODO: implement"; }

void CUDANolPDESolver::iterate(unsigned /* iterations */) {
  throw "TODO: implement";
}

void CUDANolPDESolver::cleanup() { throw "TODO: implement"; }
