#include <cstdlib>

void *cuda_malloc(size_t size);
void cuda_memcpy_host_to_device(void *device_ptr, const void *host_ptr,
                                size_t size);
void cuda_memcpy_device_to_host(void *host_ptr, const void *device_ptr,
                                size_t size);
void cuda_free(void *device_ptr);

void invoke_banded_grouping1_kernel(unsigned block_count,
                                    unsigned threads_per_block, double *x,
                                    const double *A_data,
                                    const double *sq_norms, const double *b,
                                    unsigned iterations, int bandwidth,
                                    unsigned rows_per_group,
                                    unsigned extra_rows);

void invoke_banded_grouping2_kernel(unsigned block_count,
                                    unsigned threads_per_block, double *x,
                                    const double *A_data,
                                    const double *sq_norms, const double *b,
                                    unsigned iterations, int bandwidth,
                                    unsigned rows_per_thread,
                                    unsigned extra_rows);
