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
 *     `dyscompress -rng philox` is bit-compatible with the GPU backend, and
 *   - as the oracle in the bit-compatibility test.
 */

#include <cstddef>
#include <cstdint>

namespace dyscostman {
namespace gpu {

/// Encode one AF block on the CPU, mirroring kernels K1..K8.
void cpuAfEncode(const float *interleaved, const int *ant1, const int *ant2,
                 size_t nRows, size_t nChannels, size_t nPol, size_t nAnt,
                 const float *decValues, size_t decCount, uint64_t seed,
                 uint64_t blockIndex, bool fitToMaximum, float *metaOut,
                 unsigned int *symbolsOut);

/// Decode one AF block on the CPU, mirroring kernels D1..D2 (symbols already
/// unpacked).
void cpuAfDecode(const unsigned int *symbols, const float *meta,
                 const int *ant1, const int *ant2, size_t nRows,
                 size_t nChannels, size_t nPol, size_t nAnt,
                 const float *decValues, size_t decCount,
                 float *interleavedOut);

}  // namespace gpu
}  // namespace dyscostman

#endif  // DYSCO_AF_REFERENCE_H
