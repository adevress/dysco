#ifndef DYSCO_AFTIME_BLOCK_ENCODER_H
#define DYSCO_AFTIME_BLOCK_ENCODER_H

#include "stochasticencoder.h"
#include "timeblockbuffer.h"
#include "uvector.h"

#include <complex>
#include <vector>

#include "timeblockencoder.h"

class AFTimeBlockEncoder : public TimeBlockEncoder {
 public:
  AFTimeBlockEncoder(size_t nPol, size_t nChannels, bool fitToMaximum);

  virtual ~AFTimeBlockEncoder() override;

  /**
   * Bit-compatible counterpart of the GPU encoder: same normalization order
   * as kernels K1..K6 and Philox dither as kernel K7, implemented by the
   * shared CPU reference in kernels/af_reference.cc.
   */
  virtual void EncodeWithCounterRng(
      const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
      float *metaBuffer, symbol_t *symbolBuffer, size_t antennaCount,
      uint64_t seed, uint64_t blockIndex) final override;

  virtual void InitializeDecode(const float *metaBuffer, size_t nRow,
                                size_t nAntennae) final override;

  virtual void Decode(const dyscostman::StochasticEncoder<float> &gausEncoder,
                      FBuffer &buffer, const symbol_t *symbolBuffer,
                      size_t blockRow, size_t antenna1,
                      size_t antenna2) final override;

  virtual size_t SymbolCount(size_t nRow, size_t nPol,
                             size_t nChannels) const final override {
    return nRow * nChannels * nPol * 2 /*complex*/;
  }

  virtual size_t SymbolCount(size_t nRow) const final override {
    return nRow * _nChannels * _nPol * 2 /*complex*/;
  }

  virtual size_t SymbolsPerRow() const final override {
    return _nChannels * _nPol * 2 /*complex*/;
  }

  virtual size_t MetaDataCount(size_t /*nRow*/, size_t nPol, size_t nChannels,
                               size_t nAntennae) const final override {
    return nPol * (nChannels + nAntennae);
  }

 private:
  size_t _nPol, _nChannels;
  bool _fitToMaximum;

  aocommon::UVector<double> _rmsPerChannel, _rmsPerAntenna;
};

#endif
