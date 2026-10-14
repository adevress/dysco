#ifndef DYSCO_AF_REFERENCE_H
#define DYSCO_AF_REFERENCE_H

/**
 * @file
 * Bit-exact CPU reference implementation of the GPU AF pipeline.
 *
 * The kernels in kernels/cuda/ deliberately trade some parallelism for
 * reproducibility (sequential row loops per accumulator, counter-based dither,
 * identical operation order).  This file is the host mirror of those kernels:
 * running both on the same input must produce identical metadata and
 * identical quantization symbols.  It is used
 *   - as the `EncodeWithCounterRng` implementation of the CPU encoder, so that
 *     the CPU encoder running with the same counter-based RNG is bit-compatible
 *     with the GPU backend, and
 *   - as the oracle in the bit-compatibility test.
 */

#include "span.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dyscostman::gpu {

// CPU↔GPU boundary assertion: this reference shares the buffers of the CUDA
// codec (see kernels/cuda/gpu_af_codec.h), which uses the fixed-width
// <cstdint> aliases.  The production CPU encoder feeds `symbol_t` (an
// `unsigned`) symbols; the assertion enforces the conversion is exact.
static_assert(std::is_same_v<std::uint32_t, unsigned int>);
static_assert(std::is_same_v<std::int32_t, int>);

/// Encode one AF block on the CPU, mirroring kernels K1..K8.
void cpuAfEncode(dysco::span<const float> interleaved,
                 dysco::span<const std::int32_t> ant1,
                 dysco::span<const std::int32_t> ant2,
                 std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues, std::uint64_t seed,
                 std::uint64_t blockIndex, bool fitToMaximum,
                 dysco::span<float> metaOut,
                 dysco::span<std::uint32_t> symbolsOut);

/// Decode one AF block on the CPU, mirroring kernels D1..D2 (symbols already
/// unpacked).
void cpuAfDecode(dysco::span<const std::uint32_t> symbols,
                 dysco::span<const float> meta,
                 dysco::span<const std::int32_t> ant1,
                 dysco::span<const std::int32_t> ant2, std::size_t nChannels,
                 std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues,
                 dysco::span<float> interleavedOut);

}  // namespace dyscostman::gpu

#endif  // DYSCO_AF_REFERENCE_H
