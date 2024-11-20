#ifndef CARP_UTILS_HPP
#define CARP_UTILS_HPP

void dcswp(const int *d_A_outer, const int *d_A_inner,
                     const double *d_A_values, const double *d_b,
                     const unsigned dim,
                     const double *d_sq_norms, const double *d_x, double *d_X,
                     const double relaxation, int *d_affected, const unsigned total_threads, double* d_output, const unsigned blocks);

#endif  // CARP_UTILS_HPP