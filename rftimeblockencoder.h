#ifndef DYSCO_RFTIME_BLOCK_ENCODER_H
#define DYSCO_RFTIME_BLOCK_ENCODER_H

#include "stochasticencoder.h"
#include "timeblockbuffer.h"
#include "uvector.h"

#include <complex>
#include <vector>

#include "timeblockencoder.h"

class RFTimeBlockEncoder : public TimeBlockEncoder {
 public:
  RFTimeBlockEncoder(size_t nPol, size_t nChannels);

  virtual ~RFTimeBlockEncoder() override;

  virtual void EncodeWithCounterRng(
      const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
      float *metaBuffer, symbol_t *symbolBuffer, size_t antennaCount,
      uint64_t seed, uint64_t blockIndex) final override {
    encode(gausEncoder, buffer, metaBuffer, symbolBuffer, antennaCount, seed,
           blockIndex);
  }

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

  virtual size_t MetaDataCount(size_t nRow, size_t nPol, size_t nChannels,
                               size_t /*nAntennae*/) const final override {
    return nPol * (nChannels + nRow);
  }

 private:
  /**
   * Scales the rows in the data such that every row has a value with the
   * maximum level (unless all values are 0). Polarizations within one row are
   * scaled independently, i.e. every polarization is independently maximized.
   * This function is normally called for a timeblock of data, but the data
   * array does not need to be a timeblock.
   */
  void maximizeRows(std::vector<DBufferRow> &data, float *metaBuffer,
                    double maxLevel) const;
  /**
   * Scales the channels such that every channel has a value with the maximum
   * level (unless all values are 0). Each channel consists of values with the
   * same polarizations, i.e. polarizations are scaled independently. Like @ref
   * maximizeRows() this function is normally called for one timeblock, but this
   * is not required.
   */
  void maximizeChannels(std::vector<DBufferRow> &data, float *metaBuffer,
                        double maxLevel) const;

  void encode(const dyscostman::StochasticEncoder<float> &gausEncoder,
              const FBuffer &buffer, float *metaBuffer, symbol_t *symbolBuffer,
              size_t antennaCount, uint64_t seed, uint64_t blockIndex);

  size_t _nPol, _nChannels;

  aocommon::UVector<double> _channelFactors, _rowFactors;
};

#endif
