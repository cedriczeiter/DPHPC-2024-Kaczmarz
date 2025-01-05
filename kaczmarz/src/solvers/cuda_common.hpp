#include <cstdlib>

void *cuda_malloc(size_t size);
void cuda_memcpy_host_to_device(void *device_ptr, const void *host_ptr,
                                size_t size);
void cuda_memcpy_device_to_host(void *host_ptr, const void *device_ptr,
                                size_t size);
void cuda_free(void *device_ptr);
