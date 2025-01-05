void invoke_nolpde_kernel(unsigned block_count, unsigned threads_per_block,
                          double *x, const int *A_outer, const int *A_inner,
                          const double *A_values, const double *sq_norms,
                          const double *b, unsigned iterations);
