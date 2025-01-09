#include "nolpde.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>

#include "Eigen/src/Core/PermutationMatrix.h"
#include "Eigen/src/Core/util/Constants.h"
#include "cuda_common.hpp"
#include "cuda_utils.hpp"
#include "nolpde_cuda.hpp"

void NolPDESolver::run_iterations(const Discretization& d, Vector& x,
                                  const unsigned iterations) {
  this->setup(&d, &x);
  this->iterate(iterations);
  this->flush_x();
  this->cleanup();
}

void NolPDESolver::run_iterations_with_residuals(const Discretization& d, Vector& x,
    std::vector<double>& residuals_L1,
    std::vector<double>& residuals_L2,
    std::vector<double>& residuals_Linf,
                                  const unsigned iterations) {
  this->setup(&d, &x);
  residuals_L1.push_back(this->residual_L1());
    residuals_L2.push_back(this->residual_L2());
    residuals_Linf.push_back(this->residual_Linf());
  for (unsigned iter = 0; iter < iterations; iter++) {
    this->iterate(1);
    residuals_L1.push_back(this->residual_L1());
    residuals_L2.push_back(this->residual_L2());
    residuals_Linf.push_back(this->residual_Linf());
  }
  this->flush_x();
  this->cleanup();
}

KaczmarzSolverStatus NolPDESolver::solve(const Discretization& /* lse */,
                                         Vector& /* x */,
                                         unsigned /* iterations_step */,
                                         unsigned /* max_iterations */,
                                         double /* abs_tolerance */) {
  throw "TODO: implement";
}

struct BlockFactorization {
  unsigned x_block_count, y_block_count;
};

static BlockFactorization optimal_block_factorization(
    const unsigned block_count, const unsigned max_x_block_count,
    const unsigned max_y_block_count, const double square_affinity) {
  if (block_count >= max_x_block_count * max_y_block_count) {
    // round down to multiples of two
    return {max_x_block_count & (-2), max_y_block_count & (-2)};
  }
  BlockFactorization best_factorization;
  double best_cost = std::numeric_limits<double>::infinity();

  const auto consider_factorization = [&](const unsigned x_block_count,
                                          const unsigned y_block_count) {
    if (x_block_count <= max_x_block_count &&
        y_block_count <= max_y_block_count) {
      const double non_squareness =
          std::abs(std::log((x_block_count / (double)max_x_block_count) /
                            (y_block_count / (double)max_y_block_count)));
      const double cost =
          non_squareness * square_affinity +
          (block_count - x_block_count * y_block_count) / (double)block_count;
      if (cost < best_cost) {
        best_cost = cost;
        best_factorization = {x_block_count, y_block_count};
      }
    }
  };

  for (unsigned blocks_available = block_count; blocks_available >= 1;
       blocks_available--) {
    for (unsigned div1 = 1; div1 * div1 <= blocks_available; div1++) {
      if (blocks_available % div1 == 0) {
        const unsigned div2 = blocks_available / div1;
        if (div1 % 2 == 0 && div2 % 2 == 0) {
          consider_factorization(div1, div2);
          consider_factorization(div2, div1);
        }
      }
    }
  }
  return best_factorization;
}

void PermutingNolPDESolver::setup(const Discretization* const d,
                                  Vector* const x) {
  const unsigned dim = d->position_hints.size();

  const unsigned block_count = this->get_block_count_required();

  double x_min = std::numeric_limits<double>::infinity();
  double x_max = -std::numeric_limits<double>::infinity();
  double y_min = std::numeric_limits<double>::infinity();
  double y_max = -std::numeric_limits<double>::infinity();
  for (const auto& hint : d->position_hints) {
    x_min = std::min(x_min, hint.x);
    x_max = std::max(x_max, hint.x);
    y_min = std::min(y_min, hint.y);
    y_max = std::max(y_max, hint.y);
  }

  double max_x_sep = 0.0;
  double max_y_sep = 0.0;
  const SparseMatrix& A = d->sys.A();
  for (unsigned row = 0; row < dim; row++) {
    bool hit_same = false;
    for (SparseMatrix::InnerIterator it(A, row); it; ++it) {
      if (it.col() == row) {
        hit_same = true;
      }
      const PositionHint h1 = d->position_hints[it.col()];
      const PositionHint h2 = d->position_hints[it.row()];
      max_x_sep = std::max(max_x_sep, std::abs(h1.x - h2.x));
      max_y_sep = std::max(max_y_sep, std::abs(h1.y - h2.y));
    }
    assert(hit_same);
  }

  // safety margin
  const double min_x_block_size = max_x_sep * 1.01;
  const double min_y_block_size = max_y_sep * 1.01;

  const unsigned max_x_block_count = (x_max - x_min) / min_x_block_size;
  const unsigned max_y_block_count = (y_max - y_min) / min_y_block_size;

  const BlockFactorization factorization = optimal_block_factorization(
      block_count, max_x_block_count, max_y_block_count, 1e-3);

  /*
  // uncomment if needed for debug
  std::cout << "block_count = " << block_count << std::endl;
  std::cout << "max_x_block_count = " << max_x_block_count << std::endl;
  std::cout << "max_y_block_count = " << max_y_block_count << std::endl;
  std::cout << "x_block_count = " << factorization.x_block_count << std::endl;
  std::cout << "y_block_count = " << factorization.y_block_count << std::endl;
  */

  assert(factorization.x_block_count % 2 == 0);
  assert(factorization.y_block_count % 2 == 0);

  std::vector<std::vector<std::vector<unsigned>>> blocks(
      factorization.x_block_count,
      std::vector<std::vector<unsigned>>(factorization.y_block_count));

  for (unsigned i = 0; i < dim; i++) {
    const PositionHint hint = d->position_hints[i];
    const double x_unit_coor = (hint.x - x_min) / (x_max - x_min);
    const unsigned x_block_coor =
        x_unit_coor == 1.0 ? factorization.x_block_count - 1
                           : x_unit_coor * factorization.x_block_count;
    const double y_unit_coor = (hint.y - y_min) / (y_max - y_min);
    const unsigned y_block_coor =
        y_unit_coor == 1.0 ? factorization.y_block_count - 1
                           : y_unit_coor * factorization.y_block_count;
    assert(x_block_coor < factorization.x_block_count);
    assert(y_block_coor < factorization.y_block_count);
    blocks[x_block_coor][y_block_coor].push_back(i);
  }

  this->permutation = std::vector<unsigned>(dim);
  unsigned next_in_permutation = 0;
  for (unsigned block_x = 0; block_x < factorization.x_block_count;
       block_x += 2) {
    for (unsigned block_y = 0; block_y < factorization.y_block_count;
         block_y += 2) {
      for (unsigned block_x_inner = block_x; block_x_inner < block_x + 2;
           block_x_inner++) {
        for (unsigned block_y_inner = block_y; block_y_inner < block_y + 2;
             block_y_inner++) {
          this->block_boundaries.push_back(next_in_permutation);
          for (const unsigned i : blocks[block_x_inner][block_y_inner]) {
            assert(next_in_permutation < dim);
            this->permutation[next_in_permutation++] = i;
          }
        }
      }
    }
  }
  assert(next_in_permutation == dim);
  while (this->block_boundaries.size() < block_count + 1) {
    this->block_boundaries.push_back(dim);
  }

  this->inv_permutation = std::vector<unsigned>(dim);
  for (unsigned i = 0; i < dim; i++) {
    this->inv_permutation[this->permutation[i]] = i;
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix(
      dim);
  for (unsigned i = 0; i < dim; i++) {
    permutation_matrix.indices()(i) = this->inv_permutation[i];
  }

  this->post_permute_x = permutation_matrix * *x;
  const Vector post_permute_b = permutation_matrix * d->sys.b();
  const SparseMatrix post_permute_A = permutation_matrix * d->sys.A() * permutation_matrix.transpose();
  this->post_permute_sys =
      std::make_unique<SparseLinearSystem>(post_permute_A, post_permute_b);
  this->x = x;

  // (debug) checking that the blocks that need to be non-overlapping actually
  // are non-overlapping
  assert(block_count % 4 == 0);
  for (unsigned m4 = 0; m4 < 4; m4++) {  // `m4` for "modulus mod 4"
    // for each of modulus mod 4, we require that any two rows in different
    // blocks with the same modulus are non-overlapping

    // accumulates column indices that appeared in previous blocks
    std::unordered_set<unsigned> in_prev_blocks;

    const unsigned thread_count = block_count / 4;
    for (unsigned i = 0; i < thread_count; i++) {
      const unsigned block_idx = 4 * i + m4;

      // column indicies in this block
      const auto block = [&]() {
        std::unordered_set<unsigned> block;
        const unsigned row_idx_from = this->block_boundaries[block_idx];
        const unsigned row_idx_to = this->block_boundaries[block_idx + 1];
        for (unsigned row_idx = row_idx_from; row_idx < row_idx_to; row_idx++) {
          const SparseMatrix& post_A = this->post_permute_sys->A();
          const unsigned entry_idx_from = post_A.outerIndexPtr()[row_idx];
          const unsigned entry_idx_to = post_A.outerIndexPtr()[row_idx + 1];
          for (unsigned entry_idx = entry_idx_from; entry_idx < entry_idx_to;
               entry_idx++) {
            block.insert(post_A.innerIndexPtr()[entry_idx]);
          }
        }
        return block;
      }();

      // this block may not share a column index with any of the previous blocks
      for (const unsigned e : block) {
        assert(in_prev_blocks.find(e) == in_prev_blocks.end());
      }

      // accumulate column indices for comparison with coming blocks
      in_prev_blocks.insert(block.begin(), block.end());
    }
  }

  this->post_permute_setup();
}

void PermutingNolPDESolver::flush_x() {
  this->post_permute_flush_x();

  const unsigned dim = this->post_permute_sys->b().size();
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix(
      dim);
  for (unsigned i = 0; i < dim; i++) {
    permutation_matrix.indices()(i) = this->permutation[i];
  }
  *this->x = permutation_matrix * this->post_permute_x;
}

double PermutingNolPDESolver::residual_L1() {
  this->post_permute_flush_x();
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<1>();
}

double PermutingNolPDESolver::residual_L2() {
  this->post_permute_flush_x();
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<2>();
}

double PermutingNolPDESolver::residual_Linf() {
  this->post_permute_flush_x();
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<Eigen::Infinity>();
}

unsigned CUDANolPDESolver::get_block_count_required() {
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
  this->block_boundaries_gpu = gpu_malloc_and_copy(this->block_boundaries);
}

void CUDANolPDESolver::post_permute_flush_x() {
  const size_t byte_count = this->post_permute_x.size() * sizeof(double);
  cuda_memcpy_device_to_host(&this->post_permute_x[0], this->x_gpu, byte_count);
}

void CUDANolPDESolver::iterate(unsigned iterations) {
  invoke_nolpde_kernel(this->block_count, this->threads_per_block, this->x_gpu,
                       this->A_outer_gpu, this->A_inner_gpu, this->A_values_gpu,
                       this->block_boundaries_gpu, this->sq_norms_gpu,
                       this->b_gpu, iterations);
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
  if (this->block_boundaries_gpu) {
    cuda_free(this->block_boundaries_gpu);
    this->block_boundaries_gpu = nullptr;
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

void BasicSerialNolPDESolver::setup(const Discretization* const d,
                                    Vector* const x) {
  this->d = d;
  this->x = x;
  const unsigned dim = d->sys.b().size();
  this->sq_norms = Vector(dim);
  for (unsigned i = 0; i < dim; i++) {
    const auto row = d->sys.A().row(i);
    this->sq_norms[i] = row.dot(row);
  }
}

void BasicSerialNolPDESolver::flush_x() {}

void BasicSerialNolPDESolver::iterate(const unsigned iterations) {
  const SparseMatrix& A = this->d->sys.A();
  const Vector& b = this->d->sys.b();
  const unsigned dim = b.size();
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned i = 0; i < dim; i++) {
      const auto row = A.row(i);
      const double update_coeff = (b[i] - row.dot(*x)) / sq_norms[i];
      *x += update_coeff * row;
    }
  }
}

void BasicSerialNolPDESolver::cleanup() {
  this->d = nullptr;
  this->x = nullptr;
}

double BasicSerialNolPDESolver::residual_L1() {
  return (this->d->sys.b() - this->d->sys.A() * *this->x).lpNorm<1>();
}

double BasicSerialNolPDESolver::residual_L2() {
  return (this->d->sys.b() - this->d->sys.A() * *this->x).lpNorm<2>();
}

double BasicSerialNolPDESolver::residual_Linf() {
  return (this->d->sys.b() - this->d->sys.A() * *this->x).lpNorm<Eigen::Infinity>();
}

unsigned PermutingSerialNolPDESolver::get_block_count_required() {
  return 4 * this->thread_count;
}

void PermutingSerialNolPDESolver::post_permute_setup() {
  const unsigned dim = this->post_permute_sys->b().size();
  this->sq_norms = Vector(dim);
  for (unsigned i = 0; i < dim; i++) {
    const auto row = this->post_permute_sys->A().row(i);
    sq_norms[i] = row.dot(row);
  }
}

void PermutingSerialNolPDESolver::post_permute_flush_x() {}

void PermutingSerialNolPDESolver::iterate(const unsigned iterations) {
  const SparseMatrix& A = this->post_permute_sys->A();
  const Vector& b = this->post_permute_sys->b();
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned stage = 0; stage < 4; stage++) {
      for (unsigned tid = 0; tid < this->thread_count; tid++) {
        const unsigned row_idx_from = block_boundaries[4 * tid + stage];
        const unsigned row_idx_to = block_boundaries[4 * tid + stage + 1];
        for (unsigned row_idx = row_idx_from; row_idx < row_idx_to; row_idx++) {
          const auto row = A.row(row_idx);
          const double update_coeff =
              (b[row_idx] - row.dot(this->post_permute_x)) / this->sq_norms[row_idx];
          this->post_permute_x += update_coeff * row;
        }
      }
    }
  }
}

void PermutingSerialNolPDESolver::cleanup() {}

void ShuffleSerialNolPDESolver::setup(const Discretization* const d,
                                      Vector* const x) {
  const unsigned dim = d->position_hints.size();
  this->d = d;
  this->x = x;

  this->permutation = std::vector<unsigned>(dim);
  std::iota(this->permutation.begin(), this->permutation.end(), 0);
  std::shuffle(this->permutation.begin(), this->permutation.end(),
               std::mt19937(this->shuffle_seed));

  this->inv_permutation = std::vector<unsigned>(dim);
  for (unsigned i = 0; i < dim; i++) {
    this->inv_permutation[this->permutation[i]] = i;
  }

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix(
      dim);
  for (unsigned i = 0; i < dim; i++) {
    permutation_matrix.indices()(i) = this->inv_permutation[i];
  }

  this->post_permute_x = permutation_matrix * *x;
  const Vector post_permute_b = permutation_matrix * d->sys.b();
  const SparseMatrix post_permute_A = permutation_matrix * d->sys.A() * permutation_matrix.transpose();
  this->post_permute_sys =
      std::make_unique<SparseLinearSystem>(post_permute_A, post_permute_b);

  this->sq_norms = Vector(dim);
  for (unsigned i = 0; i < dim; i++) {
    this->sq_norms[i] = post_permute_A.row(i).dot(post_permute_A.row(i));
  }
}

void ShuffleSerialNolPDESolver::flush_x() {
  const unsigned dim = this->post_permute_sys->b().size();
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix(
      dim);
  for (unsigned i = 0; i < dim; i++) {
    permutation_matrix.indices()(i) = this->permutation[i];
  }
  *this->x = permutation_matrix * this->post_permute_x;
}

void ShuffleSerialNolPDESolver::iterate(const unsigned iterations) {
  const SparseMatrix& A = this->post_permute_sys->A();
  const Vector& b = this->post_permute_sys->b();
  const unsigned dim = b.size();
  for (unsigned iter = 0; iter < iterations; iter++) {
    for (unsigned i = 0; i < dim; i++) {
      const auto row = A.row(i);
      const double update_coeff =
          (b[i] - row.dot(this->post_permute_x)) / this->sq_norms[i];
      this->post_permute_x += update_coeff * row;
    }
  }
}

void ShuffleSerialNolPDESolver::cleanup() {}

double ShuffleSerialNolPDESolver::residual_L1() {
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<1>();
}

double ShuffleSerialNolPDESolver::residual_L2() {
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<2>();
}

double ShuffleSerialNolPDESolver::residual_Linf() {
  return (this->post_permute_sys->b() - this->post_permute_sys->A() * this->post_permute_x).lpNorm<Eigen::Infinity>();
}
