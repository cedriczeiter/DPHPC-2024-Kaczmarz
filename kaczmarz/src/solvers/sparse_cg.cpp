#include "sparse_cg.hpp"

void sweep(const unsigned dim, const Vector& input, Vector& output,
           const Vector& b, const SparseLinearSystem& lse,
           const Vector& sq_norms) {
  output = input;
  // forward sweep
  for (unsigned i = 0; i < dim; i++) {
    const auto row = lse.A().row(i);
    const double update_coeff = (b[i] - row.dot(output)) / sq_norms[i];
    output += update_coeff * row;
  }
  // backward sweep
  for (unsigned i = dim; i > 0; i--) {
    const auto row = lse.A().row(i - 1);
    const double update_coeff = (b[i - 1] - row.dot(output)) / sq_norms[i - 1];
    output += update_coeff * row;
  }
}

KaczmarzSolverStatus sparse_cg(const SparseLinearSystem& lse, Vector& x,
                               const double precision,
                               const unsigned max_iterations) {
  const unsigned dim = lse.A().rows();
  Vector q = Vector::Zero(dim);
  Vector p = Vector::Zero(dim);
  Vector r = Vector::Zero(dim);
  Vector intermediate = Vector::Zero(dim);

  Vector sq_norms(dim);
  for (unsigned i = 0; i < dim; i++) {
    sq_norms[i] = lse.A().row(i).dot(lse.A().row(i));
  }

  sweep(dim, x, r, lse.b(), lse, sq_norms);
  r = r - x;
  p = r;
  for (int i = 0; i < max_iterations; i++) {
    double res = (double)((lse.A() * x - lse.b()).norm()) / lse.b().norm();
    if (res <= precision) break;
    sweep(dim, p, intermediate, Vector::Zero(dim), lse, sq_norms);
    q = p - intermediate;
    const double sq_r = r.dot(r);
    const double p_q = p.dot(q);
    const double alpha = sq_r / p_q;
    x += alpha * p;
    r += -alpha * q;
    const double beta = r.dot(r) / sq_r;
    p = r + beta * p;
  }
  return KaczmarzSolverStatus::Converged;
}