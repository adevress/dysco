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
#include <type_traits>

namespace dyscostman::gpu {

// CPU↔GPU boundary assertions.  The host callers of this API receive antenna
// indices as `int`, symbols as `unsigned int` and packed buffers as
// `unsigned char` from the storage manager (see `timeblockencoder.h`), while
// the device side uses the fixed-width <cstdint> aliases.  Callers therefore
// reinterpret_cast across this boundary; these assertions enforce that the
// conversion is exact (same width and representation).
static_assert(std::is_same_v<std::int32_t, int>);
static_assert(std::is_same_v<std::uint32_t, unsigned int>);
static_assert(std::is_same_v<std::uint8_t, unsigned char>);

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
          std::uint32_t bitsPerSymbol);
  ~AfCodec();

  AfCodec(const AfCodec &) = delete;
  AfCodec &operator=(const AfCodec &) = delete;

  /**
   * Encode one block.
   * @param interleavedIn host view, 2*nVis floats (re,im,...).
   * @param ant1,ant2     host views, nRows ints.
   * @param seed          64-bit column seed for the counter-based dither.
   * @param blockIndex    block index, part of the dither counter.
   * @param fitToMaximum  run the K6 fit-to-max optimization (parallel
   *                      damped-simultaneous "Jacobi" algorithm).
   * @param metaOut       host view, (nChannels*nPol + nPol*nAnt) floats.
   * @param packedOut     host view, ceil(2*nVis*bits/8) bytes.
   */
  void encode(dysco::span<const float> interleavedIn,
              dysco::span<const std::int32_t> ant1,
              dysco::span<const std::int32_t> ant2,
              std::uint64_t seed, std::uint64_t blockIndex, bool fitToMaximum,
              dysco::span<float> metaOut,
              dysco::span<std::uint8_t> packedOut);

  /**
   * Decode one block.
   * @param packedIn   host view with the packed symbols.
   * @param metaIn     host metadata view.
   * @param ant1,ant2  host views, nRows ints.
   * @param interleavedOut host view, 2*nVis floats.
   */
  void decode(dysco::span<const std::uint8_t> packedIn,
              dysco::span<const float> metaIn,
              dysco::span<const std::int32_t> ant1,
              dysco::span<const std::int32_t> ant2,
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
              std::uint64_t blockIndex, dysco::span<std::uint32_t> symbols);

/// Bit-pack `symbols` with kernel K8 (matches BytePacker).
void pack(dysco::span<const std::uint32_t> symbols, std::uint32_t bits,
          dysco::span<std::uint8_t> packed);

/// Bit-unpack `packed` with kernel D1 (matches BytePacker).
void unpack(dysco::span<const std::uint8_t> packed, std::uint32_t bits,
            dysco::span<std::uint32_t> symbols);

}  // namespace dyscostman::gpu

#endif  // DYSCO_GPU_AF_CODEC_H
