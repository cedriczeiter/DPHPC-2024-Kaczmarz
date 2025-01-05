#include "cuda_utils.hpp"

double* gpu_malloc_and_copy(const Vector& v) {
  /*
  const size_t byte_count = v.size() * sizeof(double);
  double* const gpu_memory = (double*)cuda_malloc(byte_count);
  cuda_memcpy_host_to_device(gpu_memory, &v[0], byte_count);
  return gpu_memory;
  */
  return gpu_malloc_and_copy(&v[0], v.size());
}
