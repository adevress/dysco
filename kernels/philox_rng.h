#ifndef DYSCO_PHILOX_RNG_H
#define DYSCO_PHILOX_RNG_H

/**
 * @file
 * Counter-based Philox4x32-10 random generator (Random123 design), implemented
 * as a header-only `__host__ __device__` component.
 *
 * The point of this generator is reproducibility: the dither value used for
 * quantizing a given symbol is a pure function of
 * `(seed, blockIndex, symbolIndex)`.  Therefore the CPU and the GPU, as well
 * as any thread scheduling, produce exactly the same dither sequence.  This is
 * what makes the GPU backend bit-compatible with the CPU backend, which is
 * impossible with `std::mt19937` (a stateful, sequential generator whose
 * output depends on thread scheduling).
 *
 * This is the *only* dither generator in Dysco (CPU and GPU): the historical
 * `std::mt19937` path has been removed.
 *
 * Stream convention (frozen; it defines re-encoding determinism):
 *   - key     = 64-bit column seed (`std::random_device` per session, or a
 *               fixed constant with `-static-seed`);
 *   - counter = (symbolIndex, blockIndex), each 64-bit, split into two 32-bit
 *               Philox counter words;
 *   - `symbolIndex = 2 * visibilityIndex + phase` with `phase` 0 for the real
 *               and 1 for the imaginary part, and `visibilityIndex` the
 *               row-major index of the visibility within the block (all rows,
 *               autocorrelations included).
 * The dither therefore depends only on the logical address of a symbol, never
 * on which thread or how many threads encode the block.
 *
 * Based on the Philox family from the Random123 library, described in:
 *
 *   John K. Salmon, Mark A. Moraes, Ron O. Dror, David E. Shaw,
 *   "Parallel Random Numbers: As Easy as 1, 2, 3",
 *   Proceedings of the International Conference for High Performance
 *   Computing, Networking, Storage and Analysis (SC'11), 2011.
 *   DOI: 10.1145/2063384.2063405
 *   Reference implementation: https://github.com/DEShawResearch/random123
 *
 * Property: Philox is a *counter-based* RNG (CBRNG).  It carries no internal
 * state; rather, it is a bijective mixing function of two inputs -- a 128-bit
 * counter and a 64-bit key -- that returns 128 bits of output.  A random
 * value is therefore *addressed* rather than drawn: the output for a given
 * (key, counter) is computed directly, in O(1), without having to generate or
 * consume any preceding values.  Three consequences are used by Dysco:
 *   1. Reproducibility: the same (seed, blockIndex, symbolIndex) always yields
 *      the same dither, independent of thread count, thread scheduling and of
 *      whether the CPU or the GPU does the encoding.
 *   2. Random access: the dither of any symbol can be produced in any order,
 *      which makes the generator trivially parallel and thread-safe -- no
 *      per-thread state, no locks, no false sharing.
 *   3. Statistical quality: the Philox4x32 permutation has a period of 2^128
 *      and passes the standard randomness test suites, so the dither is
 *      effectively white and does not bias the quantizer.
 *
 * This mirrors the Random123 `philox4x32` convention and reproduces the
 * official Random123 known-answer vectors (see tests/testphilox.cc).
 *
 * This header is intentionally kept outside `kernels/cuda/`: it is used by the
 * always-built CPU reference (`kernels/af_reference.cc`) as well as by the
 * CUDA kernel K7, so the CPU Philox path does not depend on the CUDA backend
 * being compiled or even present.  When `gpu_portability.h` is compiled by a
 * host compiler the device qualifiers expand to nothing.
 */

#include "gpu_portability.h"

#include <cstdint>

namespace dyscostman {
namespace random123 {

inline constexpr std::uint32_t kPhiloxMul0 = 0xD2511F53u;
inline constexpr std::uint32_t kPhiloxMul1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kPhiloxWeyl0 = 0x9E3779B9u;
inline constexpr std::uint32_t kPhiloxWeyl1 = 0xBB67AE85u;

DYSCO_HD DYSCO_INLINE void mulhilo32(std::uint32_t a, std::uint32_t b,
                                     std::uint32_t &hi,
                                     std::uint32_t &lo) {
  const std::uint64_t p = static_cast<std::uint64_t>(a) *
                          static_cast<std::uint64_t>(b);
  lo = static_cast<std::uint32_t>(p);
  hi = static_cast<std::uint32_t>(p >> 32);
}

DYSCO_HD DYSCO_INLINE void philox4x32round(std::uint32_t &c0,
                                           std::uint32_t &c1,
                                           std::uint32_t &c2,
                                           std::uint32_t &c3,
                                           std::uint32_t k0,
                                           std::uint32_t k1) {
  std::uint32_t hi0, lo0, hi1, lo1;
  mulhilo32(kPhiloxMul0, c0, hi0, lo0);
  mulhilo32(kPhiloxMul1, c2, hi1, lo1);
  const std::uint32_t n0 = hi1 ^ c1 ^ k0;
  const std::uint32_t n1 = lo1;
  const std::uint32_t n2 = hi0 ^ c3 ^ k1;
  const std::uint32_t n3 = lo0;
  c0 = n0;
  c1 = n1;
  c2 = n2;
  c3 = n3;
}

/// Philox4x32 with 10 rounds.  The counter and key are passed by value and the
/// resulting random numbers are returned through the counter variables.
DYSCO_HD DYSCO_INLINE void philox4x32_10(std::uint32_t &c0, std::uint32_t &c1,
                                         std::uint32_t &c2, std::uint32_t &c3,
                                         std::uint32_t k0, std::uint32_t k1) {
  for (std::int32_t round = 0; round != 10; ++round) {
    philox4x32round(c0, c1, c2, c3, k0, k1);
    if (round != 9) {
      k0 += kPhiloxWeyl0;
      k1 += kPhiloxWeyl1;
    }
  }
}

/**
 * Produce the dither value for a symbol.  The result is in [0, 2^31), matching
 * the range of the legacy `std::uniform_int_distribution<unsigned>(0,
 * (1u<<31)-1)` used by the CPU encoder.
 *
 * @param seedLo,seedHi 64-bit column seed (split in two 32-bit halves).
 * @param blockIndex    Index of the time block within the column.
 * @param symbolIndex   Index of the symbol within the block.
 */
[[nodiscard]] DYSCO_HD DYSCO_INLINE std::uint32_t ditherValue(
    std::uint32_t seedLo, std::uint32_t seedHi, std::uint64_t blockIndex,
    std::uint64_t symbolIndex) {
  std::uint32_t c0 = static_cast<std::uint32_t>(symbolIndex);
  std::uint32_t c1 = static_cast<std::uint32_t>(symbolIndex >> 32);
  std::uint32_t c2 = static_cast<std::uint32_t>(blockIndex);
  std::uint32_t c3 = static_cast<std::uint32_t>(blockIndex >> 32);
  philox4x32_10(c0, c1, c2, c3, seedLo, seedHi);
  return c0 & 0x7fffffffu;
}

/**
 * Dither value for one phase (0 = real, 1 = imaginary) of the visibility at
 * @p visibilityIndex within @p blockIndex.  This is the single place where the
 * frozen symbol indexing convention is applied, shared by every encoder and by
 * kernel K7.
 */
[[nodiscard]] DYSCO_HD DYSCO_INLINE std::uint32_t ditherForVisibility(
    std::uint32_t seedLo, std::uint32_t seedHi, std::uint64_t blockIndex,
    std::uint64_t visibilityIndex, unsigned phase) {
  return ditherValue(seedLo, seedHi, blockIndex,
                     visibilityIndex * 2 + phase);
}

}  // namespace random123
}  // namespace dyscostman

#endif  // DYSCO_PHILOX_RNG_H
