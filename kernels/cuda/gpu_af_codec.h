#ifndef DYSCO_GPU_AF_CODEC_H
#define DYSCO_GPU_AF_CODEC_H

/**
 * @file
 * Public interface of the CUDA AF codec.  This header does not include any
 * CUDA header, so it can be included by regular C++ translation units
 * (`dyscodatacolumn.cc`, the CLI tools, the unit tests) when the CUDA backend
 * is enabled.
 *
 * Buffers are passed as `dysco::span` views (which are `std::span` under
 * C++20).  All host/device transfers go through CUDA managed memory
 * (transparent page migration).
 */

#include "../span.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace dyscostman::gpu {

/// Whether a usable GPU device is present at runtime.
[[nodiscard]] bool available();

/// Human readable device name (or "none").
[[nodiscard]] std::string_view deviceName();

/**
 * Codec for one AF-normalized time block.  One instance is owned by each
 * encoder thread / decoder, so the managed workspace is reused between blocks.
 */
class AfCodec {
 public:
  AfCodec(std::size_t nRows, std::size_t nChannels, std::size_t nPol,
          std::size_t nAnt, dysco::span<const float> decValues,
          unsigned bitsPerSymbol);
  ~AfCodec();

  AfCodec(const AfCodec &) = delete;
  AfCodec &operator=(const AfCodec &) = delete;

  /**
   * Encode one block.
   * @param interleavedIn host view, 2*nVis floats (re,im,...).
   * @param ant1,ant2     host views, nRows ints.
   * @param seed          64-bit column seed for the Philox dither.
   * @param blockIndex    block index, part of the dither counter.
   * @param fitToMaximum  run the K6 fit-to-max optimization (parallel
   *                      damped-simultaneous "Jacobi" algorithm).
   * @param metaOut       host view, (nChannels*nPol + nPol*nAnt) floats.
   * @param packedOut     host view, ceil(2*nVis*bits/8) bytes.
   */
  void encode(dysco::span<const float> interleavedIn,
              dysco::span<const int> ant1, dysco::span<const int> ant2,
              std::uint64_t seed, std::uint64_t blockIndex, bool fitToMaximum,
              dysco::span<float> metaOut,
              dysco::span<unsigned char> packedOut);

  /**
   * Decode one block.
   * @param packedIn   host view with the packed symbols.
   * @param metaIn     host metadata view.
   * @param ant1,ant2  host views, nRows ints.
   * @param interleavedOut host view, 2*nVis floats.
   */
  void decode(dysco::span<const unsigned char> packedIn,
              dysco::span<const float> metaIn, dysco::span<const int> ant1,
              dysco::span<const int> ant2,
              dysco::span<float> interleavedOut);

  [[nodiscard]] std::size_t packedByteCount() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> _impl;
};

// --------------------------------------------------------------------------
// Small helpers exposing individual kernels, used by the unit tests.  They
// run on the default stream and synchronize before returning.
// --------------------------------------------------------------------------

/// Quantize `values` with kernel K7 using the counter-based dither.
/// `symbols` must have the same size as `values`.
void quantize(dysco::span<const float> values,
              dysco::span<const float> decValues, std::uint64_t seed,
              std::uint64_t blockIndex, dysco::span<unsigned int> symbols);

/// Bit-pack `symbols` with kernel K8 (matches BytePacker).
void pack(dysco::span<const unsigned int> symbols, unsigned bits,
          dysco::span<unsigned char> packed);

/// Bit-unpack `packed` with kernel D1 (matches BytePacker).
void unpack(dysco::span<const unsigned char> packed, unsigned bits,
            dysco::span<unsigned int> symbols);

}  // namespace dyscostman::gpu

#endif  // DYSCO_GPU_AF_CODEC_H
