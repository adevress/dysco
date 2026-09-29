#ifndef DYSCO_GPU_PORTABILITY_H
#define DYSCO_GPU_PORTABILITY_H

/**
 * @file
 * Thin portability layer that isolates the GPU-specific vocabulary.
 *
 * The Dysco GPU backend is split in one source file per kernel, and those
 * files only use the macros declared here (plus the small runtime helpers).
 * Today only the CUDA backend is implemented; the kernel bodies are written
 * in an index-based, vendor-neutral style so a HIP or SYCL port is a matter
 * of adding a branch here (and mapping `DYSCO_LAUNCH_1D`), rather than
 * touching every kernel.
 *
 * Backends:
 *   - DYSCO_USE_CUDA (or compilation by nvcc): NVIDIA CUDA.
 *   - DYSCO_USE_HIP: reserved.  Not implemented yet; it is a hard error so a
 *     future port cannot be silently mistaken for a working build.
 *   - none: plain host compilation.  The device qualifiers become empty,
 *     which keeps the headers parseable by a regular C++ compiler.
 */

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#if defined(DYSCO_USE_CUDA) || defined(__CUDACC__)

#include <cuda_runtime.h>

#define DYSCO_HD __host__ __device__
#define DYSCO_GLOBAL __global__
#define DYSCO_INLINE __forceinline__
typedef cudaStream_t dysco_stream_t;
#define DYSCO_DEFAULT_STREAM 0

namespace dysco_gpu {

[[nodiscard]] inline std::string_view device_name() {
  static char name[256] = "none";
  int device = 0;
  if (cudaGetDevice(&device) == cudaSuccess) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess)
      std::snprintf(name, sizeof(name), "%s", properties.name);
  }
  return std::string_view(name);
}

/// Deleter for managed buffers held by std::unique_ptr.
struct ManagedDeleter {
  void operator()(void* p) const noexcept {
    if (p != nullptr) cudaFree(p);
  }
};

}  // namespace dysco_gpu

#define DYSCO_GPU_CHECK(call)                                             \
  do {                                                                    \
    const cudaError_t error = (call);                                     \
    if (error != cudaSuccess) {                                           \
      throw std::runtime_error(std::string("Dysco GPU error: ") +         \
                               cudaGetErrorString(error) + " (" #call ")"); \
    }                                                                     \
  } while (false)

#define DYSCO_MANAGED_ALLOC(ptr, bytes) \
  DYSCO_GPU_CHECK(cudaMallocManaged((ptr), (bytes)))
#define DYSCO_FREE(ptr) DYSCO_GPU_CHECK(cudaFree(ptr))

#elif defined(DYSCO_USE_HIP)

// Reserved for a future HIP port.  The kernels already avoid vendor types,
// but the backend still needs a hip_runtime include, hipStream_t,
// hipMallocManaged/hipFree, a hipError_t checker and a mapping of
// DYSCO_LAUNCH_1D.  Fail loudly instead of producing a broken build.
#error "Dysco: the HIP backend is not implemented yet; only CUDA (ENABLE_CUDA) is supported."

#else

// Host-only compilation.  Device qualifiers vanish; the runtime helper is only
// declared (its definition lives in the accelerator translation unit) to avoid
// an ODR clash with the accelerator definition.
#define DYSCO_HD
#define DYSCO_GLOBAL
#define DYSCO_INLINE inline
typedef void* dysco_stream_t;
#define DYSCO_DEFAULT_STREAM 0

namespace dysco_gpu {
[[nodiscard]] std::string_view device_name();
}  // namespace dysco_gpu

#define DYSCO_GPU_CHECK(call) (void)(call)
#define DYSCO_MANAGED_ALLOC(ptr, bytes) ((ptr) = nullptr)
#define DYSCO_FREE(ptr) ((void)(ptr))

#endif

/// Launch a one-dimensional kernel with a stream (CUDA syntax; a HIP/SYCL
/// port maps this macro instead of touching the launchers).
#define DYSCO_LAUNCH_1D(kernel, blocks, threads, stream, ...) \
  do {                                                        \
    kernel<<<(blocks), (threads), 0, (stream)>>>(__VA_ARGS__); \
  } while (false)

#endif  // DYSCO_GPU_PORTABILITY_H
