#include "../kaczmarz_serial/kaczmarz_common.hpp"
#include "linear_systems/sparse.hpp"

KaczmarzSolverStatus banded_sparse_kaczmarz(const BandedLinearSystem& lse, Eigen::VectorXd& x, unsigned max_iterations, double precision);
