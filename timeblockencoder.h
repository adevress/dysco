#ifndef DYSCO_TIME_BLOCK_ENCODER_H
#define DYSCO_TIME_BLOCK_ENCODER_H

#include "stochasticencoder.h"
#include "timeblockbuffer.h"
#include "uvector.h"

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <vector>

class RMSMeasurement {
 public:
  RMSMeasurement() : _count(0), _value(0.0) {}

  void Include(const std::complex<double> &val) {
    if (isfinite(val)) {
      _count++;
      _value += val.real() * val.real() + val.imag() * val.imag();
    }
  }

  double RMS() const { return sqrt(_value / (_count * 2)); }

 private:
  static bool isfinite(const std::complex<double> &val) {
    return std::isfinite(val.real()) && std::isfinite(val.imag());
  }

  size_t _count;
  double _value;
};

class TimeBlockEncoder {
 public:
  typedef TimeBlockBuffer<std::complex<float>> FBuffer;
  typedef typename TimeBlockBuffer<std::complex<float>>::DataRow FBufferRow;
  typedef TimeBlockBuffer<std::complex<double>> DBuffer;
  typedef typename TimeBlockBuffer<std::complex<double>>::DataRow DBufferRow;

  typedef unsigned symbol_t;

  virtual ~TimeBlockEncoder() = default;

  virtual void EncodeWithoutDithering(
      const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
      float *metaBuffer, symbol_t *symbolBuffer, size_t antennaCount) = 0;

  /**
   * Dithered encoding.  The dither is a pure function of
   * `(seed, blockIndex, symbolIndex)` through the counter-based Philox
   * generator (see kernels/philox_rng.h), so the result is independent of
   * thread scheduling and of the number of threads, and is bit-identical
   * between the CPU and GPU encoders.
   */
  virtual void EncodeWithCounterRng(
      const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
      float *metaBuffer, symbol_t *symbolBuffer, size_t antennaCount,
      uint64_t seed, uint64_t blockIndex) = 0;

  /**
   * Optional fast path that encodes *and* bit-packs the block on an
   * accelerator.  Returns true when @p packed was filled (number of bytes in
   * @p packedBytes).  The default returns false, in which case the caller
   * runs EncodeWithCounterRng() + BytePacker::pack().
   */
  virtual bool EncodePacked(
      const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
      float *metaBuffer, unsigned char *packed, size_t antennaCount,
      uint64_t seed, uint64_t blockIndex, size_t &packedBytes) {
    (void)gausEncoder;
    (void)buffer;
    (void)metaBuffer;
    (void)packed;
    (void)antennaCount;
    (void)seed;
    (void)blockIndex;
    (void)packedBytes;
    return false;
  }

  /**
   * Decode a whole *packed* block at once, without a prior host unpack step.
   * Accelerators override this.  Returns true when @p buffer was filled.
   */
  virtual bool DecodePacked(const dyscostman::StochasticEncoder<float> &,
                            FBuffer &, const unsigned char *, size_t, const int *,
                            const int *, size_t, size_t, const float *) {
    return false;
  }

  /**
   * Decode a whole block at once.  The default implementation loops over the
   * rows calling Decode(); accelerators override this to decode in bulk.
   * InitializeDecode() must have been called before.
   */
  virtual void DecodeBlock(const dyscostman::StochasticEncoder<float> &gausEncoder,
                           FBuffer &buffer, const symbol_t *symbolBuffer,
                           const int *ant1, const int *ant2, size_t nRows,
                           size_t /*nAntennae*/, const float * /*metaBuffer*/) {
    for (size_t row = 0; row != nRows; ++row)
      Decode(gausEncoder, buffer, symbolBuffer, row,
             static_cast<size_t>(ant1[row]), static_cast<size_t>(ant2[row]));
  }

  virtual void InitializeDecode(const float *metaBuffer, size_t nRow,
                                size_t nAntennae) = 0;

  virtual void Decode(const dyscostman::StochasticEncoder<float> &gausEncoder,
                      FBuffer &buffer, const symbol_t *symbolBuffer,
                      size_t blockRow, size_t antenna1, size_t antenna2) = 0;

  virtual size_t SymbolCount(size_t nRow, size_t nPol,
                             size_t nChannels) const = 0;

  virtual size_t SymbolCount(size_t nRow) const = 0;

  virtual size_t SymbolsPerRow() const = 0;

  virtual size_t MetaDataCount(size_t nRow, size_t nPol, size_t nChannels,
                               size_t nAntennae) const = 0;

 protected:
  TimeBlockEncoder() = default;
};

#endif
