#ifndef DYSCO_CUDA_CCCL_CUH
#define DYSCO_CUDA_CCCL_CUH

/**
 * @file
 * Small collection of CUDA C++ helpers shared by the device kernels.
 *
 * These helpers are deliberately thin wrappers around CCCL (libcu++ / CUB),
 * which the project requires at the CUDA 12 level (either the CCCL bundled
 * with CUDA >= 12, or the standalone CCCL package):
 *
 *   - `cuda::std::span` / `cuda::std::mdspan` for typed, bounds-carrying
 *     device views (the kernels no longer take raw pointer/size pairs),
 *   - `cuda::atomic_ref` (libcu++) for scoped atomic maximum operations,
 *   - `cub::BlockReduce` + `cuda::maximum<>` for order-independent block-wide
 *     reductions,
 *   - `cuda::std` math in place of the global-namespace intrinsics.
 *
 * Keeping the CUDA-specific vocabulary in one place makes the individual
 * kernels express their algorithms rather than their CUDA plumbing, and makes
 * a future HIP/SYCL backend a matter of replacing this header.
 */

#include "../gpu_portability.h"

#include <cuda/atomic>
#include <cuda/functional>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/mdspan>
#include <cuda/std/span>

namespace dyscostman {
namespace gpu {

/// Row-major two-dimensional device view, backed by `cuda::std::mdspan`.
/// All multi-dimensional arrays in this backend are contiguous and row-major
/// (the default `layout_right`), so a single alias is enough.
template <class T>
using MdSpan2D =
    cuda::std::mdspan<T, cuda::std::dextents<cuda::std::size_t, 2>>;

/// Component measure used by the encoder: `max(|re|, |im|)`.
///
/// Warning: cuda::max differs from std::max for Nan
[[nodiscard]] DYSCO_HD DYSCO_INLINE double componentMax(double re,
                                                        double im) noexcept {
  return cuda::std::fmax(cuda::std::fabs(re), cuda::std::fabs(im));
}

/// Quiet NaN with the canonical IEEE-754 bit pattern (0x7ff8...), identical to
/// what `std::numeric_limits<double>::quiet_NaN()` produces on the CPU side.
/// A compile-time constant instead of a `nan("")` call.
inline constexpr cuda::std::uint64_t kQuietNaNBits = 0x7ff8000000000000ull;

/// Reinterpret a *non-negative* double as an unsigned 64-bit key.  The IEEE-754
/// bit patterns of non-negative doubles are order-isomorphic to unsigned
/// integers, so an integer `fetch_max` computes a double maximum.
[[nodiscard]] DYSCO_HD DYSCO_INLINE constexpr cuda::std::uint64_t
doubleToMaxKey(double value) noexcept {
  return cuda::std::bit_cast<cuda::std::uint64_t>(value);
}

[[nodiscard]] DYSCO_HD DYSCO_INLINE constexpr double maxKeyToDouble(
    cuda::std::uint64_t key) noexcept {
  return cuda::std::bit_cast<double>(key);
}

/// Flattened index of this thread in a 1-D grid; the loop entry of a
/// grid-stride loop whose stride is `gridStrideSize()`.
[[nodiscard]] DYSCO_HD DYSCO_INLINE std::size_t threadFlattenedId() noexcept {
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

/// Total number of threads in a 1-D grid; the stride of a grid-stride loop.
[[nodiscard]] DYSCO_HD DYSCO_INLINE std::size_t gridStrideSize() noexcept {
  return static_cast<std::size_t>(gridDim.x) * blockDim.x;
}

/// Block-scoped atomic maximum of non-negative double keys.  On shared memory
/// libcu++ lowers this to the native shared atomic when the architecture
/// supports it and to a CAS loop otherwise.
DYSCO_HD DYSCO_INLINE void atomicMaxKeyBlock(
    cuda::std::uint64_t *address, cuda::std::uint64_t key) noexcept {
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_block> ref(*address);
  ref.fetch_max(key, cuda::memory_order_relaxed);
}

/// Device-scoped atomic maximum for a non-negative double.
DYSCO_HD DYSCO_INLINE void atomicMaxNonNegative(double *address,
                                                double value) noexcept {
  cuda::atomic_ref<cuda::std::uint64_t, cuda::thread_scope_device> ref(
      *reinterpret_cast<cuda::std::uint64_t *>(address));
  ref.fetch_max(doubleToMaxKey(value), cuda::memory_order_relaxed);
}

/// Device-scoped atomic OR, used for the K6 progress / early-exit flags.
DYSCO_HD DYSCO_INLINE void atomicOrDevice(std::int32_t *address,
                                          std::int32_t value) noexcept {
  cuda::atomic_ref<std::int32_t, cuda::thread_scope_device> ref(*address);
  ref.fetch_or(value, cuda::memory_order_relaxed);
}

}  // namespace gpu
}  // namespace dyscostman

#endif  // DYSCO_CUDA_CCCL_CUH
