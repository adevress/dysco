#ifndef DYSCO_ISPC_AF_TIME_BLOCK_ENCODER_H
#define DYSCO_ISPC_AF_TIME_BLOCK_ENCODER_H

#include "timeblockencoder.h"

#include <memory>
#include <vector>

namespace dyscostman {

namespace ispc {
class IspcAfCodec;
}  // namespace ispc

/**
 * AF time-block encoder backed by the ISPC SIMD kernels.  It is created by
 * DyscoDataColumn when the ISPC backend is selected.  The bit stream it
 * produces is identical to `AFTimeBlockEncoder::EncodeWithCounterRng` (CPU
 * Philox) for the no-fit-to-max path, and uses the same damped-simultaneous
 * fit-to-max algorithm as the CUDA backend otherwise.
 */
class IspcAFTimeBlockEncoder final : public TimeBlockEncoder {
 public:
  IspcAFTimeBlockEncoder(size_t nPol, size_t nChannels,
                         std::vector<float> decValues, unsigned bits,
                         bool fitToMaximum);
  ~IspcAFTimeBlockEncoder() override;

  void EncodeWithCounterRng(const dyscostman::StochasticEncoder<float> &,
                            FBuffer &, float *, symbol_t *, size_t, uint64_t,
                            uint64_t) final override {
    throw std::runtime_error(
        "IspcAFTimeBlockEncoder only supports the packed fast path");
  }
  bool EncodePacked(const dyscostman::StochasticEncoder<float> &, FBuffer &,
                    float *, unsigned char *, size_t, uint64_t, uint64_t,
                    size_t &) final override;

  void InitializeDecode(const float *metaBuffer, size_t nRow,
                        size_t nAntennae) final override;
  void Decode(const dyscostman::StochasticEncoder<float> &, FBuffer &,
              const symbol_t *, size_t, size_t, size_t) final override {
    throw std::runtime_error(
        "IspcAFTimeBlockEncoder decodes whole blocks, not rows");
  }
  bool DecodePacked(const dyscostman::StochasticEncoder<float> &, FBuffer &,
                    const unsigned char *, size_t, const int *, const int *,
                    size_t, size_t, const float *) final override;

  size_t SymbolCount(size_t nRow, size_t nPol,
                     size_t nChannels) const final override {
    return nRow * nChannels * nPol * 2;
  }
  size_t SymbolCount(size_t nRow) const final override {
    return nRow * _nChannels * _nPol * 2;
  }
  size_t SymbolsPerRow() const final override {
    return _nChannels * _nPol * 2;
  }
  size_t MetaDataCount(size_t, size_t nPol, size_t nChannels,
                       size_t nAntennae) const final override {
    return nPol * (nChannels + nAntennae);
  }

 private:
  void flatten(FBuffer &buffer, std::vector<float> &interleaved,
               std::vector<int> &ant1, std::vector<int> &ant2) const;
  void ensureCodec(size_t nRows, size_t nAnt);

  size_t _nPol, _nChannels;
  size_t _codecRows = 0, _codecAnt = 0;
  std::vector<float> _decValues;
  unsigned _bits;
  bool _fitToMaximum;
  std::unique_ptr<ispc::IspcAfCodec> _codec;
};

}  // namespace dyscostman

#endif  // DYSCO_ISPC_AF_TIME_BLOCK_ENCODER_H
