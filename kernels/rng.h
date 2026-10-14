#ifndef DYSCO_RNG_H
#define DYSCO_RNG_H

/**
 * @file
 * Counter-based dither RNG facade shared by the CPU and the GPU encoders.
 *
 * The dither value applied when quantizing a symbol is a pure function of
 * `(seed, blockIndex, symbolIndex)`, which is what makes the encoders
 * reproducible: the result does not depend on the thread scheduling, on the
 * number of threads, nor on whether the CPU or the GPU does the encoding.
 *
 * The generator is the Threefish-based `threefry` CBRNG from the alea library
 * (https://github.com/adevress/alea), used through its `threefry4x64`
 * specialisation.
 *
 * Stream convention (frozen; it defines re-encoding determinism):
 *   - key     = 64-bit column seed (`std::random_device` per session, or a
 *               fixed constant with `-static-seed`);
 *   - counter = (symbolIndex, blockIndex), each 64-bit;
 *   - `symbolIndex = 2 * visibilityIndex + phase` with `phase` 0 for the real
 *     and 1 for the imaginary part, and `visibilityIndex` the row-major index
 *     of the visibility within the block (all rows, autocorrelations
 *     included).
 *
 * Threefry counter/key mapping: the 4x64 generator is keyed with
 * `{seed, 0, 0, 0}` and takes the counter `{symbolIndex, blockIndex, 0, 0}`.
 * Only the low 31 bits of the first output word are used, matching the dither
 * domain `[0, 2^31)` of the quantizer
 * (`StochasticEncoder::EncodeWithDithering`).
 */

#include "gpu_portability.h"

#include <alea/threefry.hpp>

#include <cstdint>

namespace dyscostman {
namespace rng {

/// Human-readable name of the compiled-in dither generator.
inline constexpr const char *kName = "threefry";

/// alea's 4x64 Threefish-based CBRNG, the dither generator of the library.
using threefry_type = alea::threefry4x64;

/**
 * Produce the dither value for a symbol.  The result is in [0, 2^31),
 * matching the range of the legacy `std::uniform_int_distribution<unsigned>(0,
 * (1u<<31)-1)` used by the CPU encoder.
 *
 * @param seedLo,seedHi 64-bit column seed (split in two 32-bit halves).
 * @param blockIndex    Index of the time block within the column.
 * @param symbolIndex   Index of the symbol within the block.
 */
[[nodiscard]] DYSCO_HD DYSCO_INLINE std::uint32_t ditherValue(
    std::uint32_t seedLo, std::uint32_t seedHi, std::uint64_t blockIndex,
    std::uint64_t symbolIndex) {
  const std::uint64_t seed =
      (static_cast<std::uint64_t>(seedHi) << 32) | seedLo;
  const typename threefry_type::key_type key{seed, 0, 0, 0};
  const typename threefry_type::domain_type counter{symbolIndex, blockIndex, 0,
                                                    0};
  const threefry_type generator(key);
  const typename threefry_type::range_type block = generator(counter);
  return static_cast<std::uint32_t>(block[0] & 0x7fffffffu);
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
  return ditherValue(seedLo, seedHi, blockIndex, visibilityIndex * 2 + phase);
}

}  // namespace rng
}  // namespace dyscostman

#endif  // DYSCO_RNG_H
