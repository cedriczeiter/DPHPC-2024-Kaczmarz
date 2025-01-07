void invoke_nolpde_kernel(unsigned block_count, unsigned threads_per_block,
                          double *x, const unsigned *A_outer,
                          const unsigned *A_inner, const double *A_values,
                          const unsigned *block_boundaries,
                          const double *sq_norms, const double *b,
                          unsigned iterations);
