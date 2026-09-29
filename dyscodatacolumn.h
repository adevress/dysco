#ifndef DYSCO_DATA_COLUMN_H
#define DYSCO_DATA_COLUMN_H

#include "threadeddyscocolumn.h"

#include "stochasticencoder.h"
#include "timeblockencoder.h"

namespace dyscostman {

class DyscoStMan;

/**
 * A column for storing compressed complex values with an approximate Gaussian
 * distribution.
 * @author André Offringa
 */
class DyscoDataColumn final : public ThreadedDyscoColumn<std::complex<float>> {
 public:
  /**
   * Create a new column. Internally called by DyscoStMan when creating a
   * new column.
   */
  DyscoDataColumn(DyscoStMan *parent, int dtype)
      : ThreadedDyscoColumn(parent, dtype),
        _philoxSeed(std::random_device{}()),
        _gausEncoder(),
        _distribution(GaussianDistribution),
        _normalization(Normalization::kRF),
        _studentsTNu(0.0),
        _gpuEnabled(false),
        _fitToMaximum(true) {}

  DyscoDataColumn(const DyscoDataColumn &source) = delete;

  void operator=(const DyscoDataColumn &source) = delete;

  /** Destructor. */
  virtual ~DyscoDataColumn() { shutdown(); }

  virtual void Prepare(DyscoDistribution distribution,
                       Normalization normalization, double studentsTNu,
                       double distributionTruncation) override;

  void SetStaticRandomizationSeed() {
    std::cout
        << "Warning: Initializing random number generator with static seed!\n";
    _philoxSeed = 0x9E3779B97F4A7C15ull;
  }

  /// Enable the GPU encoder/decoder for this column (requires ENABLE_CUDA).
  void SetGpuEnabled(bool enabled) { _gpuEnabled = enabled; }

  /// Enable the AF fit-to-maximum post-normalization.
  void SetFitToMaximum(bool enabled) { _fitToMaximum = enabled; }

 protected:
  virtual void initializeDecode(TimeBlockBuffer<data_t> *buffer,
                                const float *metaBuffer, size_t nRow,
                                size_t nAntennae) override;

  virtual void decode(TimeBlockBuffer<data_t> *buffer, const symbol_t *data,
                      size_t blockRow, size_t a1, size_t a2) override;

  virtual std::unique_ptr<ThreadDataBase> initializeEncodeThread() override;

  virtual void encode(ThreadDataBase *threadData,
                      TimeBlockBuffer<data_t> *buffer, float *metaBuffer,
                      symbol_t *symbolBuffer, size_t nAntennae,
                      uint64_t blockIndex) override;

  virtual size_t encodePacked(ThreadDataBase *threadData,
                              TimeBlockBuffer<data_t> *buffer,
                              float *metaBuffer, unsigned char *packed,
                              size_t nAntennae, uint64_t blockIndex) override;

  virtual bool decodePacked(TimeBlockBuffer<data_t> *buffer,
                            const unsigned char *packed, size_t packedBytes,
                            const int *ant1, const int *ant2, size_t nRows,
                            size_t nAntennae, const float *meta) override;

  virtual void decodeBlock(TimeBlockBuffer<data_t> *buffer,
                           const symbol_t *symbolBuffer, const int *ant1,
                           const int *ant2, size_t nRows, size_t nAntennae,
                           const float *meta) override;

  virtual size_t metaDataFloatCount(size_t nRow, size_t nPolarizations,
                                    size_t nChannels,
                                    size_t nAntennae) const override;

  virtual size_t symbolCount(size_t nRowsInBlock, size_t nPolarizations,
                             size_t nChannels) const override;

 private:
  struct ThreadData final : public ThreadDataBase {
    ThreadData(std::unique_ptr<TimeBlockEncoder> timeBlockEncoder)
        : encoder(std::move(timeBlockEncoder)) {}
    std::unique_ptr<TimeBlockEncoder> encoder;
  };

  uint64_t _philoxSeed;
  std::unique_ptr<StochasticEncoder<float>> _gausEncoder;
  std::unique_ptr<TimeBlockEncoder> _decoder;
  DyscoDistribution _distribution;
  Normalization _normalization;
  double _studentsTNu;
  bool _gpuEnabled;
  bool _fitToMaximum;
};

}  // namespace dyscostman

#endif
