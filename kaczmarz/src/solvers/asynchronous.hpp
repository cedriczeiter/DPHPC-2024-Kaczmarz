#include <unistd.h>

#include "common.hpp"
#include "linear_systems/sparse.hpp"
#include "linear_systems/types.hpp"

KaczmarzSolverStatus asynchronous_gpu(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads);


KaczmarzSolverStatus asynchronous_cpu(const SparseLinearSystem &lse,
                                              Vector &x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const unsigned num_threads);