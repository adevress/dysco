#ifndef DYSCO_GPU_AF_TIME_BLOCK_ENCODER_H
#define DYSCO_GPU_AF_TIME_BLOCK_ENCODER_H

#include "timeblockencoder.h"

#include <memory>
#include <vector>

namespace dyscostman {

namespace gpu {
class AfCodec;
}  // namespace gpu

/**
 * AF time-block encoder backed by the CUDA kernels.  It is created by
 * DyscoDataColumn when the CUDA backend is selected.  The bit stream it
 * produces is identical to `AFTimeBlockEncoder::EncodeWithCounterRng`, i.e.
 * to the CPU encoder running with the Philox counter RNG.
 */
class GpuAFTimeBlockEncoder final : public TimeBlockEncoder {
 public:
  GpuAFTimeBlockEncoder(size_t nPol, size_t nChannels,
                        std::vector<float> decValues, unsigned bits,
                        bool fitToMaximum);
  ~GpuAFTimeBlockEncoder() override;

  void EncodeWithCounterRng(const dyscostman::StochasticEncoder<float> &,
                            FBuffer &, float *, symbol_t *, size_t, uint64_t,
                            uint64_t) final override {
    throw std::runtime_error(
        "GpuAFTimeBlockEncoder only supports the packed fast path");
  }
  bool EncodePacked(const dyscostman::StochasticEncoder<float> &, FBuffer &,
                    float *, unsigned char *, size_t, uint64_t, uint64_t,
                    size_t &) final override;

  void InitializeDecode(const float *metaBuffer, size_t nRow,
                        size_t nAntennae) final override;
  void Decode(const dyscostman::StochasticEncoder<float> &, FBuffer &,
              const symbol_t *, size_t, size_t, size_t) final override {
    throw std::runtime_error(
        "GpuAFTimeBlockEncoder decodes whole blocks, not rows");
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
  std::unique_ptr<gpu::AfCodec> _codec;
};

}  // namespace dyscostman

#endif  // DYSCO_GPU_AF_TIME_BLOCK_ENCODER_H
