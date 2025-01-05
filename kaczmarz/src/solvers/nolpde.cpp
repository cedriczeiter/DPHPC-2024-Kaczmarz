#include "nolpde.hpp"

#include "cuda_common.hpp"
#include "cuda_utils.hpp"
#include "nolpde_cuda.hpp"

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

unsigned CUDANolPDESolver::get_blocks_required() {
  return this->block_count * this->threads_per_block * 4;
}

void CUDANolPDESolver::post_permute_setup() {
  this->cleanup();
  this->x_gpu = gpu_malloc_and_copy(this->post_permute_x);
  const unsigned dim = this->post_permute_x.size();
  {
    std::vector<double> sq_norms(this->post_permute_x.size());
    for (unsigned row_idx = 0; row_idx < dim; row_idx++) {
      const Eigen::SparseVector<double> row =
          this->post_permute_sys->A().innerVector(row_idx);
      sq_norms[row_idx] = row.dot(row);
    }
    this->sq_norms_gpu = gpu_malloc_and_copy(sq_norms);
  }
  this->b_gpu = gpu_malloc_and_copy(this->post_permute_sys->b());
  const auto& A = this->post_permute_sys->A();
  this->A_outer_gpu =
      (unsigned*)gpu_malloc_and_copy(A.outerIndexPtr(), dim + 1);
  this->A_inner_gpu =
      (unsigned*)gpu_malloc_and_copy(A.innerIndexPtr(), A.nonZeros());
  this->A_values_gpu = gpu_malloc_and_copy(A.valuePtr(), A.nonZeros());
}

void CUDANolPDESolver::post_permute_flush_x() {
  const size_t byte_count = this->post_permute_x.size() * sizeof(double);
  cuda_memcpy_device_to_host(&this->post_permute_x[0], this->x_gpu, byte_count);
}

void CUDANolPDESolver::iterate(unsigned iterations) {
  invoke_nolpde_kernel(this->block_count, this->threads_per_block, this->x_gpu,
                       this->A_outer_gpu, this->A_inner_gpu, this->A_values_gpu,
                       this->sq_norms_gpu, this->b_gpu, iterations);
}

void CUDANolPDESolver::cleanup() {
  if (this->x_gpu) {
    cuda_free(this->x_gpu);
    this->x_gpu = nullptr;
  }
  if (this->A_outer_gpu) {
    cuda_free(this->A_outer_gpu);
    this->A_outer_gpu = nullptr;
  }
  if (this->A_inner_gpu) {
    cuda_free(this->A_inner_gpu);
    this->A_inner_gpu = nullptr;
  }
  if (this->A_values_gpu) {
    cuda_free(this->A_values_gpu);
    this->A_values_gpu = nullptr;
  }
  if (this->sq_norms_gpu) {
    cuda_free(this->sq_norms_gpu);
    this->sq_norms_gpu = nullptr;
  }
  if (this->b_gpu) {
    cuda_free(this->b_gpu);
    this->b_gpu = nullptr;
  }
}
