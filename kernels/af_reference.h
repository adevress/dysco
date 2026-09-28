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
 *     the CPU encoder running with the Philox RNG is bit-compatible with the
 *     GPU backend, and
 *   - as the oracle in the bit-compatibility test.
 */

#include "span.h"

#include <cstddef>
#include <cstdint>

namespace dyscostman::gpu {

/// Encode one AF block on the CPU, mirroring kernels K1..K8.
void cpuAfEncode(dysco::span<const float> interleaved,
                 dysco::span<const int> ant1, dysco::span<const int> ant2,
                 std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues, std::uint64_t seed,
                 std::uint64_t blockIndex, bool fitToMaximum,
                 dysco::span<float> metaOut,
                 dysco::span<unsigned int> symbolsOut);

/// Decode one AF block on the CPU, mirroring kernels D1..D2 (symbols already
/// unpacked).
void cpuAfDecode(dysco::span<const unsigned int> symbols,
                 dysco::span<const float> meta, dysco::span<const int> ant1,
                 dysco::span<const int> ant2, std::size_t nChannels,
                 std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues,
                 dysco::span<float> interleavedOut);

}  // namespace dyscostman::gpu

#endif  // DYSCO_AF_REFERENCE_H
