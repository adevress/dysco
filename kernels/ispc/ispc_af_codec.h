#ifndef DYSCO_ISPC_AF_CODEC_H
#define DYSCO_ISPC_AF_CODEC_H

/**
 * @file
 * Public interface of the ISPC AF codec.  Like `kernels/cuda/gpu_af_codec.h`,
 * this header does not include any ISPC-generated header, so it can be
 * included by regular C++ translation units (`dyscodatacolumn.cc`, the CLI
 * tools, the unit tests).
 *
 * Buffers are passed as `dysco::span` views.  Unlike the CUDA backend the
 * workspace is ordinary host memory (ISPC is a CPU SIMD backend), so there is
 * no managed-memory / transfer machinery: the codec simply reuses `std::vector`
 * buffers between blocks.
 */

#include "../span.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>

namespace dyscostman {
namespace ispc {

// CPU<->ISPC boundary assertions.  The generated ISPC headers use the
// fixed-width <cstdint> aliases; the rest of Dysco hands over `int`,
// `unsigned` and `unsigned char` from the storage manager (see
// `timeblockencoder.h`).  The assertions enforce that the conversions are
// exact.
static_assert(std::is_same_v<std::int32_t, int>);
static_assert(std::is_same_v<std::uint32_t, unsigned int>);
static_assert(std::is_same_v<std::uint8_t, unsigned char>);

/// Whether an ISPC codec is usable at runtime (always true when compiled in).
[[nodiscard]] bool available();

/// Human readable backend / target name.
[[nodiscard]] std::string_view deviceName();

/**
 * Codec for one AF-normalized time block, backed by the ISPC kernels.  One
 * instance is owned by each encoder thread / decoder; the workspace vectors
 * are reused between blocks.
 *
 * The encoded bit stream matches the CPU encoder running with the Philox
 * counter RNG (and therefore the CUDA backend) for the no-fit-to-max path;
 * with fit-to-max it uses the same damped-simultaneous algorithm as the CUDA
 * backend.
 */
class IspcAfCodec {
 public:
  IspcAfCodec(std::size_t nRows, std::size_t nChannels, std::size_t nPol,
              std::size_t nAnt, dysco::span<const float> decValues,
              std::uint32_t bitsPerSymbol);
  ~IspcAfCodec();

  IspcAfCodec(const IspcAfCodec &) = delete;
  IspcAfCodec &operator=(const IspcAfCodec &) = delete;

  void encode(dysco::span<const float> interleavedIn,
              dysco::span<const std::int32_t> ant1,
              dysco::span<const std::int32_t> ant2, std::uint64_t seed,
              std::uint64_t blockIndex, bool fitToMaximum,
              dysco::span<float> metaOut,
              dysco::span<std::uint8_t> packedOut);

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

// Small helpers exposing individual kernels, used by the unit tests.  They
// mirror the CUDA helpers of the same name.
void quantize(dysco::span<const float> values,
              dysco::span<const float> decValues, std::uint64_t seed,
              std::uint64_t blockIndex, dysco::span<std::uint32_t> symbols);

void pack(dysco::span<const std::uint32_t> symbols, std::uint32_t bits,
          dysco::span<std::uint8_t> packed);

void unpack(dysco::span<const std::uint8_t> packed, std::uint32_t bits,
            dysco::span<std::uint32_t> symbols);

}  // namespace ispc
}  // namespace dyscostman

#endif  // DYSCO_ISPC_AF_CODEC_H
