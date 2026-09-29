#include "gpu_af_time_block_encoder.h"

#ifdef DYSCO_HAS_CUDA

#include "kernels/cuda/gpu_af_codec.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

namespace dyscostman {

GpuAFTimeBlockEncoder::GpuAFTimeBlockEncoder(size_t nPol, size_t nChannels,
                                             std::vector<float> decValues,
                                             unsigned bits, bool fitToMaximum)
    : _nPol(nPol),
      _nChannels(nChannels),
      _decValues(std::move(decValues)),
      _bits(bits),
      _fitToMaximum(fitToMaximum) {}

void GpuAFTimeBlockEncoder::ensureCodec(size_t nRows, size_t nAnt) {
  if (_codec && _codecRows == nRows && _codecAnt == nAnt) return;
  _codec = std::make_unique<gpu::AfCodec>(nRows, _nChannels, _nPol, nAnt,
                                          _decValues, _bits);
  _codecRows = nRows;
  _codecAnt = nAnt;
}

GpuAFTimeBlockEncoder::~GpuAFTimeBlockEncoder() = default;

void GpuAFTimeBlockEncoder::flatten(FBuffer &buffer,
                                    std::vector<float> &interleaved,
                                    std::vector<std::int32_t> &ant1,
                                    std::vector<std::int32_t> &ant2) const {
  const auto &rows = buffer.GetVector();
  const size_t nRows = rows.size();
  const size_t visPerRow = _nPol * _nChannels;
  interleaved.resize(nRows * visPerRow * 2);
  ant1.resize(nRows);
  ant2.resize(nRows);
  for (size_t r = 0; r != nRows; ++r) {
    // Boundary cast: casacore antenna indices are size_t; the codec stores
    // fixed-width std::int32_t (antenna counts fit by construction).
    ant1[r] = static_cast<std::int32_t>(rows[r].antenna1);
    ant2[r] = static_cast<std::int32_t>(rows[r].antenna2);
    std::memcpy(interleaved.data() + r * visPerRow * 2,
                rows[r].visibilities.data(), visPerRow * 2 * sizeof(float));
  }
}

bool GpuAFTimeBlockEncoder::EncodePacked(
    const dyscostman::StochasticEncoder<float> &, FBuffer &buffer,
    float *metaBuffer, unsigned char *packed, size_t antennaCount,
    uint64_t seed, uint64_t blockIndex, size_t &packedBytes) {
  const auto &rows = buffer.GetVector();
  ensureCodec(rows.size(), antennaCount);
  std::vector<float> interleaved;
  std::vector<std::int32_t> ant1, ant2;
  flatten(buffer, interleaved, ant1, ant2);
  const std::size_t metaCount = _nChannels * _nPol + _nPol * antennaCount;
  _codec->encode(
      interleaved, ant1, ant2, seed, blockIndex, _fitToMaximum,
      dysco::span<float>(metaBuffer, metaCount),
      dysco::span<std::uint8_t>(reinterpret_cast<std::uint8_t *>(packed),
                                _codec->packedByteCount()));
  packedBytes = _codec->packedByteCount();
  return true;
}

void GpuAFTimeBlockEncoder::InitializeDecode(const float *, size_t, size_t) {
  // The GPU decoder reads the metadata directly; nothing to precompute.
}

bool GpuAFTimeBlockEncoder::DecodePacked(
    const dyscostman::StochasticEncoder<float> &, FBuffer &buffer,
    const unsigned char *packed, size_t packedBytes, const int *ant1,
    const int *ant2, size_t nRows, size_t nAntennae, const float *meta) {
  ensureCodec(nRows, nAntennae);
  const std::size_t visPerRow = _nPol * _nChannels;
  const std::size_t nVis = nRows * visPerRow;
  const std::size_t metaCount = _nChannels * _nPol + _nPol * nAntennae;
  std::vector<float> interleaved(2 * nVis);
  // Boundary casts: the storage manager hands over `unsigned char` packed
  // bytes and `int` antenna indices (see timeblockencoder.h); the codec API
  // uses the fixed-width aliases.  The asserts enforce the casts are exact.
  static_assert(std::is_same_v<std::uint8_t, unsigned char>);
  static_assert(std::is_same_v<std::int32_t, int>);
  _codec->decode(
      dysco::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t *>(packed),
                                      packedBytes),
      dysco::span<const float>(meta, metaCount),
      dysco::span<const std::int32_t>(reinterpret_cast<const std::int32_t *>(ant1),
                                      nRows),
      dysco::span<const std::int32_t>(reinterpret_cast<const std::int32_t *>(ant2),
                                      nRows),
      interleaved);
  buffer.resize(nRows);
  for (size_t r = 0; r != nRows; ++r) {
    FBufferRow &row = buffer[r];
    row.antenna1 = static_cast<size_t>(ant1[r]);
    row.antenna2 = static_cast<size_t>(ant2[r]);
    row.visibilities.resize(visPerRow);
    for (size_t i = 0; i != visPerRow; ++i)
      row.visibilities[i] =
          std::complex<float>(interleaved[(r * visPerRow + i) * 2],
                              interleaved[(r * visPerRow + i) * 2 + 1]);
  }
  return true;
}

}  // namespace dyscostman

#endif  // DYSCO_HAS_CUDA
