#ifndef DYSCO_GPU_AF_CODEC_H
#define DYSCO_GPU_AF_CODEC_H

/**
 * @file
 * Public interface of the CUDA AF codec.  This header does not include any
 * CUDA header, so it can be included by regular C++ translation units
 * (`dyscodatacolumn.cc`, the CLI tools) when the CUDA backend is enabled.
 *
 * All host/device transfers go through CUDA managed memory (transparent page
 * migration): the workspace buffers are allocated with `cudaMallocManaged`,
 * filled/read from the host, and migrated on demand by the kernel launches.
 */

#include <cstddef>
#include <cstdint>
#include <memory>

namespace dyscostman {
namespace gpu {

/// Whether a usable GPU device is present at runtime.
bool available();

/// Human readable device name (or "none").
const char *deviceName();

/**
 * Codec for one AF-normalized time block.  One instance is owned by each
 * encoder thread / decoder, so the managed workspace is reused between blocks.
 */
class AfCodec {
 public:
  AfCodec(size_t nRows, size_t nChannels, size_t nPol, size_t nAnt,
          const float *decValues, size_t decCount, unsigned bitsPerSymbol);
  ~AfCodec();

  AfCodec(const AfCodec &) = delete;
  AfCodec &operator=(const AfCodec &) = delete;

  /**
   * Encode one block.
   * @param interleavedIn  host buffer, 2*nVis floats (re,im,...).
   * @param ant1,ant2      host arrays, nRows ints.
   * @param seed           64-bit column seed for the Philox dither.
   * @param blockIndex     block index, part of the dither counter.
   * @param fitToMaximum   run the K6 greedy optimization.
   * @param metaOut        host buffer, (nChannels*nPol + nPol*nAnt) floats.
   * @param packedOut      host buffer, ceil(2*nVis*bits/8) bytes.
   */
  void encode(const float *interleavedIn, const int *ant1, const int *ant2,
              uint64_t seed, uint64_t blockIndex, bool fitToMaximum,
              float *metaOut, unsigned char *packedOut);

  /**
   * Decode one block.
   * @param packedIn   host buffer with the packed symbols.
   * @param packedBytes number of packed bytes.
   * @param metaIn     host metadata buffer.
   * @param ant1,ant2  host arrays, nRows ints.
   * @param interleavedOut host buffer, 2*nVis floats.
   */
  void decode(const unsigned char *packedIn, size_t packedBytes,
              const float *metaIn, const int *ant1, const int *ant2,
              float *interleavedOut);

  size_t packedByteCount() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

}  // namespace gpu
}  // namespace dyscostman

#endif  // DYSCO_GPU_AF_CODEC_H
