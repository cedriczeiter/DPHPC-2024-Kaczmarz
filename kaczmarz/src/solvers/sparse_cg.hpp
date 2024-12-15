#include <chrono>
#include <climits>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

KaczmarzSolverStatus sparse_cg(const SparseLinearSystem& lse, Vector& x,
                               const double precision,
                               const unsigned max_iterations);