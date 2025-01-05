#include <cstdlib>
#include <vector>

#include "cuda_common.hpp"
#include "linear_systems/types.hpp"

template <typename T>
T* gpu_malloc_and_copy(const T* v, const size_t item_count) {
  const size_t byte_count = item_count * sizeof(T);
  T* const gpu_memory = (T*)cuda_malloc(byte_count);
  cuda_memcpy_host_to_device(gpu_memory, v, byte_count);
  return gpu_memory;
}

template <typename T>
T* gpu_malloc_and_copy(const std::vector<T>& v) {
  return gpu_malloc_and_copy(&v[0], v.size());
  /*
  const size_t byte_count = v.size() * sizeof(T);
  T* const gpu_memory = (T*)cuda_malloc(byte_count);
  cuda_memcpy_host_to_device(gpu_memory, &v[0], byte_count);
  return gpu_memory;
  */
}

double* gpu_malloc_and_copy(const Vector& v);
