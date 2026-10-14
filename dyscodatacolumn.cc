#include "dyscodatacolumn.h"
#include "aftimeblockencoder.h"
#include "rftimeblockencoder.h"
#include "rowtimeblockencoder.h"

#ifdef DYSCO_HAS_CUDA
#include "gpu_af_time_block_encoder.h"
#endif

namespace dyscostman {

void DyscoDataColumn::Prepare(DyscoDistribution distribution,
                              Normalization normalization, double studentsTNu,
                              double distributionTruncation) {
  _normalization = normalization;
  ThreadedDyscoColumn::Prepare(distribution, normalization, studentsTNu,
                               distributionTruncation);
  const size_t nPolarizations = shape()[0], nChannels = shape()[1];

  switch (distribution) {
    case GaussianDistribution:
      _gausEncoder.reset(
          new StochasticEncoder<float>(1 << getBitsPerSymbol(), 1.0, true));
      break;
    case UniformDistribution:
      _gausEncoder.reset(
          new StochasticEncoder<float>(1 << getBitsPerSymbol(), 1.0, false));
      break;
    case StudentsTDistribution:
      _gausEncoder.reset(new StochasticEncoder<float>(
          StochasticEncoder<float>::StudentTEncoder(1 << getBitsPerSymbol(),
                                                    studentsTNu, 1.0)));
      break;
    case TruncatedGaussianDistribution:
      _gausEncoder.reset(new StochasticEncoder<float>(
          StochasticEncoder<float>::TruncatedGausEncoder(
              1 << getBitsPerSymbol(), distributionTruncation, 1.0)));
      break;
  }

  bool useGpu = false;
#ifdef DYSCO_HAS_CUDA
  useGpu = _gpuEnabled && normalization == Normalization::kAF;
#endif
  if (useGpu) {
#ifdef DYSCO_HAS_CUDA
    _decoder.reset(new GpuAFTimeBlockEncoder(
        nPolarizations, nChannels, _gausEncoder->DecodeValues(),
        getBitsPerSymbol(), _fitToMaximum));
#endif
  } else {
    switch (normalization) {
      case Normalization::kAF:
        _decoder.reset(
            new AFTimeBlockEncoder(nPolarizations, nChannels, _fitToMaximum));
        break;
      case Normalization::kRF:
        _decoder.reset(new RFTimeBlockEncoder(nPolarizations, nChannels));
        break;
      case Normalization::kRow:
        _decoder.reset(new RowTimeBlockEncoder(nPolarizations, nChannels));
        break;
    }
  }
}

void DyscoDataColumn::initializeDecode(TimeBlockBuffer<data_t> * /*buffer*/,
                                       const float *metaBuffer, size_t nRow,
                                       size_t nAntennae) {
  _decoder->InitializeDecode(metaBuffer, nRow, nAntennae);
}

void DyscoDataColumn::decode(TimeBlockBuffer<data_t> *buffer,
                             const unsigned int *data, size_t blockRow,
                             size_t a1, size_t a2) {
  _decoder->Decode(*_gausEncoder, *buffer, data, blockRow, a1, a2);
}

void DyscoDataColumn::decodeBlock(TimeBlockBuffer<data_t> *buffer,
                                  const symbol_t *symbolBuffer,
                                  const int *ant1, const int *ant2, size_t nRows,
                                  size_t nAntennae, const float *meta) {
  _decoder->DecodeBlock(*_gausEncoder, *buffer, symbolBuffer, ant1, ant2, nRows,
                        nAntennae, meta);
}

bool DyscoDataColumn::decodePacked(TimeBlockBuffer<data_t> *buffer,
                                   const unsigned char *packed,
                                   size_t packedBytes, const int *ant1,
                                   const int *ant2, size_t nRows,
                                   size_t nAntennae, const float *meta) {
  if (!_decoder) return false;
  return _decoder->DecodePacked(*_gausEncoder, *buffer, packed, packedBytes,
                                ant1, ant2, nRows, nAntennae, meta);
}

std::unique_ptr<ThreadedDyscoColumn<std::complex<float>>::ThreadDataBase>
DyscoDataColumn::initializeEncodeThread() {
  const size_t nPolarizations = shape()[0], nChannels = shape()[1];
  std::unique_ptr<TimeBlockEncoder> encoder;
  bool useGpu = false;
#ifdef DYSCO_HAS_CUDA
  useGpu = _gpuEnabled && _normalization == Normalization::kAF;
#endif
  if (useGpu) {
#ifdef DYSCO_HAS_CUDA
    encoder.reset(new GpuAFTimeBlockEncoder(
        nPolarizations, nChannels, _gausEncoder->DecodeValues(),
        getBitsPerSymbol(), _fitToMaximum));
#endif
  } else {
    switch (_normalization) {
      case Normalization::kAF:
        encoder.reset(
            new AFTimeBlockEncoder(nPolarizations, nChannels, _fitToMaximum));
        break;
      case Normalization::kRF:
        encoder.reset(new RFTimeBlockEncoder(nPolarizations, nChannels));
        break;
      case Normalization::kRow:
        encoder.reset(new RowTimeBlockEncoder(nPolarizations, nChannels));
        break;
    }
  }
  // No per-thread RNG state: the dither is a pure function of the symbol's
  // logical address (see kernels/rng.h).
  return std::make_unique<ThreadData>(std::move(encoder));
}

void DyscoDataColumn::encode(ThreadDataBase *threadData,
                             TimeBlockBuffer<data_t> *buffer, float *metaBuffer,
                             symbol_t *symbolBuffer, size_t nAntennae,
                             uint64_t blockIndex) {
  ThreadData &data = static_cast<ThreadData &>(*threadData);
  data.encoder->EncodeWithCounterRng(*_gausEncoder, *buffer, metaBuffer,
                                     symbolBuffer, nAntennae, _ditherSeed,
                                     blockIndex);
}

size_t DyscoDataColumn::encodePacked(ThreadDataBase *threadData,
                                     TimeBlockBuffer<data_t> *buffer,
                                     float *metaBuffer, unsigned char *packed,
                                     size_t nAntennae, uint64_t blockIndex) {
  ThreadData &data = static_cast<ThreadData &>(*threadData);
  size_t packedBytes = 0;
  if (data.encoder->EncodePacked(*_gausEncoder, *buffer, metaBuffer, packed,
                                 nAntennae, _ditherSeed, blockIndex,
                                 packedBytes))
    return packedBytes;
  return 0;
}

size_t DyscoDataColumn::metaDataFloatCount(size_t nRows, size_t nPolarizations,
                                           size_t nChannels,
                                           size_t nAntennae) const {
  return _decoder->MetaDataCount(nRows, nPolarizations, nChannels, nAntennae);
}

size_t DyscoDataColumn::symbolCount(size_t nRowsInBlock, size_t nPolarizations,
                                    size_t nChannels) const {
  return _decoder->SymbolCount(nRowsInBlock, nPolarizations, nChannels);
}

}  // namespace dyscostman
