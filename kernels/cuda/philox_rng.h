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
 * Reference: Salmon, Moraes, Dror, Shaw, "Parallel Random Numbers: As Easy as
 * 1, 2, 3" (2011).  This mirrors the Random123 `philox4x32` convention.
 */

#include "../gpu_portability.h"

#include <cstdint>

namespace dyscostman {
namespace random123 {

inline constexpr std::uint32_t kPhiloxMul0 = 0xD2511F53u;
inline constexpr std::uint32_t kPhiloxMul1 = 0xCD9E8D57u;
inline constexpr std::uint32_t kPhiloxWeys0 = 0x9E3779B9u;
inline constexpr std::uint32_t kPhiloxWeys1 = 0xBB67AE85u;

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
  for (int round = 0; round != 10; ++round) {
    philox4x32round(c0, c1, c2, c3, k0, k1);
    if (round != 9) {
      k0 += kPhiloxWeys0;
      k1 += kPhiloxWeys1;
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

}  // namespace random123
}  // namespace dyscostman

#endif  // DYSCO_PHILOX_RNG_H
