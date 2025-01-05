#include "cuda_common.hpp"

void *cuda_malloc(const size_t size) {
  void *ptr;
  cudaMalloc(&ptr, size);
  return ptr;
}

void cuda_memcpy_host_to_device(void *const device_ptr,
                                const void *const host_ptr, const size_t size) {
  cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
}

void cuda_memcpy_device_to_host(void *const host_ptr,
                                const void *const device_ptr,
                                const size_t size) {
  cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
}

void cuda_free(void *const device_ptr) { cudaFree(device_ptr); }
