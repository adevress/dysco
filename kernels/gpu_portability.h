#ifndef DYSCO_GPU_PORTABILITY_H
#define DYSCO_GPU_PORTABILITY_H

/**
 * @file
 * Thin portability layer that isolates the GPU vendor-specific vocabularies.
 *
 * The Dysco GPU backend is split in one source file per kernel. Those files
 * should only use the macros declared here and the small runtime helpers, so
 * that porting the kernels to HIP or SYCL is a matter of adding a new branch
 * in this header (and a mapping of the launch helper), rather than touching
 * every kernel.
 *
 * Supported backends:
 *   - DYSCO_USE_CUDA : NVIDIA CUDA (default when ENABLE_CUDA=ON and nvcc is used)
 *   - DYSCO_USE_HIP  : AMD HIP (placeholder, same API surface)
 *   - (none)         : plain host compilation; DYSCO_HD / DYSCO_DEV become empty
 *
 * SYCL mapping note: the SYCL equivalent of a CUDA kernel is a
 * `parallel_for` over a `sycl::queue`.  The kernel bodies below are written in
 * an index-based style (no CUDA-specific types inside the computation) so the
 * body can be reused inside a SYCL handler with `id<1>` / `id<2>` indices.
 */

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#if defined(DYSCO_USE_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>

#define DYSCO_HD __host__ __device__
#define DYSCO_DEV __device__
#define DYSCO_GLOBAL __global__
#define DYSCO_INLINE __forceinline__
#define DYSCO_SHARED __shared__
typedef cudaStream_t dysco_stream_t;
#define DYSCO_DEFAULT_STREAM 0

namespace dysco_gpu {
inline bool available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}
inline const char *device_name() {
  static char name[256] = "none";
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return name;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
    snprintf(name, sizeof(name), "%s", prop.name);
  return name;
}
inline void synchronize() { cudaDeviceSynchronize(); }
}  // namespace dysco_gpu

#define DYSCO_GPU_CHECK(call)                                              \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      throw std::runtime_error(std::string("Dysco GPU error: ") +          \
                               cudaGetErrorString(err) + " (" #call ")");  \
    }                                                                      \
  } while (false)

#define DYSCO_MANAGED_ALLOC(ptr, bytes) DYSCO_GPU_CHECK(cudaMallocManaged((ptr), (bytes)))
#define DYSCO_FREE(ptr) DYSCO_GPU_CHECK(cudaFree(ptr))
#define DYSCO_PREFETCH(ptr, bytes, dest) \
  cudaMemPrefetchAsync((ptr), (bytes), (dest), DYSCO_DEFAULT_STREAM)
#define DYSCO_MEMCPY(dst, src, bytes) \
  DYSCO_GPU_CHECK(cudaMemcpy((dst), (src), (bytes), cudaMemcpyDefault))

#elif defined(DYSCO_USE_HIP)
#include <hip/hip_runtime.h>

#define DYSCO_HD __host__ __device__
#define DYSCO_DEV __device__
#define DYSCO_GLOBAL __global__
#define DYSCO_INLINE __forceinline__
#define DYSCO_SHARED __shared__
typedef hipStream_t dysco_stream_t;
#define DYSCO_DEFAULT_STREAM 0

namespace dysco_gpu {
inline bool available() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
inline const char *device_name() {
  static char name[256] = "none";
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) return name;
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) == hipSuccess)
    snprintf(name, sizeof(name), "%s", prop.name);
  return name;
}
inline void synchronize() { hipDeviceSynchronize(); }
}  // namespace dysco_gpu

#define DYSCO_GPU_CHECK(call)                                              \
  do {                                                                     \
    hipError_t err = (call);                                               \
    if (err != hipSuccess) {                                               \
      throw std::runtime_error(std::string("Dysco GPU error: ") +          \
                               hipGetErrorString(err) + " (" #call ")");   \
    }                                                                      \
  } while (false)

#define DYSCO_MANAGED_ALLOC(ptr, bytes) DYSCO_GPU_CHECK(hipMallocManaged((ptr), (bytes)))
#define DYSCO_FREE(ptr) DYSCO_GPU_CHECK(hipFree(ptr))
#define DYSCO_PREFETCH(ptr, bytes, dest) hipMemPrefetchAsync((ptr), (bytes), (dest))
#define DYSCO_MEMCPY(dst, src, bytes) \
  DYSCO_GPU_CHECK(hipMemcpy((dst), (src), (bytes), hipMemcpyDefault))

#else
// Host-only compilation: the macros vanish, which makes the kernel headers
// parseable by the regular C++ compiler (useful for the CPU fallback and for
// static analysis).
#define DYSCO_HD
#define DYSCO_DEV
#define DYSCO_GLOBAL
#define DYSCO_INLINE inline
#define DYSCO_SHARED
typedef void *dysco_stream_t;
#define DYSCO_DEFAULT_STREAM 0

// Declarations only.  The definitions live in the CUDA/HIP translation unit
// (gpu_af_codec.cu).  Defining them here as host stubs would create an ODR
// clash with the accelerator definitions and the linker could silently pick
// the stub.
namespace dysco_gpu {
bool available();
const char *device_name();
void synchronize();
}  // namespace dysco_gpu

#define DYSCO_GPU_CHECK(call) (void)(call)
#define DYSCO_MANAGED_ALLOC(ptr, bytes) ((ptr) = nullptr)
#define DYSCO_FREE(ptr) ((void)(ptr))
#define DYSCO_PREFETCH(ptr, bytes, dest) ((void)(ptr))
#define DYSCO_MEMCPY(dst, src, bytes) ((void)(dst))

#endif

/// Launch a one-dimensional kernel with `count` work-items.
#define DYSCO_LAUNCH_1D(kernel, blocks, threads, stream, ...) \
  kernel<<<(blocks), (threads), 0, (stream)>>>(__VA_ARGS__)

#endif  // DYSCO_GPU_PORTABILITY_H
