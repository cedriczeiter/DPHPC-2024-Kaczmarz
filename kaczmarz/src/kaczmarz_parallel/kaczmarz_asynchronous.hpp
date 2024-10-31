#include <omp.h>
#include <unistd.h>

#include "../kaczmarz_serial/kaczmarz_common.hpp"
#include "linear_systems/sparse.hpp"

KaczmarzSolverStatus sparse_kaczmarz_parallel(const SparseLinearSystem &lse,
                                              double *x,
                                              const unsigned max_iterations,
                                              const double precision,
                                              const double num_threads);
