#ifndef CARP_UTILS_HPP
#define CARP_UTILS_HPP

#define L_RESIDUAL 1000
#define ROWS_PER_THREAD 1
#define LOCAL_RUNS_PER_THREAD 1
#define THREADS_PER_BLOCK 256


void dcswp(const int *d_A_outer, const int *d_A_inner,
                     const double *d_A_values, const double *d_b,
                     const unsigned dim,
                     const double *d_sq_norms, const double *d_x, double *d_X,
                     const double relaxation, int *d_affected, const unsigned total_threads, double* d_output, const unsigned blocks);

void add_gpu(const double* d_a, const double* d_b, double* d_output, const double factor, const unsigned dim);
void copy_gpu(const double* d_from, double* d_to, const unsigned dim);
double dot_product_gpu(const double* d_a, const double* d_b, double *d_to, const unsigned dim);

#endif  // CARP_UTILS_HPP