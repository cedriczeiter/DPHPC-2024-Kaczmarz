#include <chrono>
#include <cmath>
#include <iostream>
#include "common.hpp"
#include "linear_systems/sparse.hpp"

#include <chrono>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>
#include "linear_systems/types.hpp"

#include <chrono>
#include <climits>

KaczmarzSolverStatus sparse_cg(const SparseLinearSystem& lse, Vector& x, const double precision, const unsigned max_iterations);