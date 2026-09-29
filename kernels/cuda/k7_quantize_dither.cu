// K7 — dithered dictionary quantization.
//
// Reproduces `StochasticEncoder<float>::EncodeWithDithering` for every value
// of the block.  The binary search is performed over the *decode* values (the
// centroids), exactly like the CPU encoder.  The dither value comes from the
// counter-based Philox generator, indexed by the global symbol index, so the
// sequence is independent of how the work is distributed over threads.

#include "cuda_kernels.h"
#include "philox_rng.h"

namespace dyscostman {
namespace gpu {

namespace {

/// Half the 32-bit dither range: matches the CPU
/// `std::uniform_int_distribution<unsigned>(0, (1u<<31)-1)` dither domain.
constexpr float kDitherRange = 2147483648.0f;

[[nodiscard]] DYSCO_HD DYSCO_INLINE unsigned int quantizeDither(
    float value, cuda::std::span<const float> decValues,
    unsigned int dither) {
  const std::size_t decCount = decValues.size();
  if (!cuda::std::isfinite(value)) return static_cast<unsigned int>(decCount);
  // lower_bound over decValues: first index with decValues[index] >= value.
  std::size_t lo = 0, hi = decCount;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) >> 1;
    if (decValues[mid] < value)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0) return 0;
  if (lo >= decCount) return static_cast<unsigned int>(decCount - 1);
  const float rightValue = decValues[lo];
  const float leftValue = decValues[lo - 1];
  const float ditherMark =
      kDitherRange * (value - leftValue) / (rightValue - leftValue);
  return ditherMark > static_cast<float>(dither)
             ? static_cast<unsigned int>(lo)
             : static_cast<unsigned int>(lo - 1);
}

// Flat variant used by the tests: one symbol per input value, with the dither
// indexed by the value index (the AF path uses 2*i / 2*i+1).
__global__ void quantizeFlatKernel(cuda::std::span<const float> values,
                                   cuda::std::span<const float> decValues,
                                   uint32_t seedLo, uint32_t seedHi,
                                   uint64_t blockIndex,
                                   cuda::std::span<unsigned int> symbols) {
  const std::size_t i = threadFlattenedId();
  if (i >= values.size()) return;
  const unsigned int dither =
      random123::ditherValue(seedLo, seedHi, blockIndex, i);
  symbols[i] = quantizeDither(values[i], decValues, dither);
}

__global__ void quantizeDitherKernel(MdSpan2D<const double> re,
                                     MdSpan2D<const double> im,
                                     cuda::std::span<const float> decValues,
                                     uint32_t seedLo, uint32_t seedHi,
                                     uint64_t blockIndex,
                                     cuda::std::span<unsigned int> symbols) {
  const std::size_t visPerRow = re.extent(1);
  const std::size_t nVis = re.extent(0) * visPerRow;
  const std::size_t i = threadFlattenedId();
  if (i >= nVis) return;
  const std::size_t row = i / visPerRow;
  const std::size_t vi = i - row * visPerRow;
  const uint64_t base = static_cast<uint64_t>(i) * 2;
  const unsigned int dRe = random123::ditherValue(seedLo, seedHi, blockIndex,
                                                  base);
  const unsigned int dIm = random123::ditherValue(seedLo, seedHi, blockIndex,
                                                  base + 1);
  symbols[base] =
      quantizeDither(static_cast<float>(re(row, vi)), decValues, dRe);
  symbols[base + 1] =
      quantizeDither(static_cast<float>(im(row, vi)), decValues, dIm);
}

}  // namespace

void k7_quantize_flat(cuda::std::span<const float> values,
                      cuda::std::span<const float> decValues,
                      std::uint32_t seedLo, std::uint32_t seedHi,
                      std::uint64_t blockIndex,
                      cuda::std::span<unsigned int> symbols,
                      dysco_stream_t stream) {
  constexpr unsigned kThreads = kThreadBlockSize;
  const std::size_t count = values.size();
  const auto blocks = gridFor(count, kThreads);
  DYSCO_LAUNCH_1D(quantizeFlatKernel, blocks, kThreads, stream, values,
                  decValues, seedLo, seedHi, blockIndex, symbols);
}

void k7_quantize_dither(MdSpan2D<const double> re, MdSpan2D<const double> im,
                        cuda::std::span<const float> decValues,
                        std::uint32_t seedLo, std::uint32_t seedHi,
                        std::uint64_t blockIndex,
                        cuda::std::span<unsigned int> symbols,
                        dysco_stream_t stream) {
  constexpr unsigned kThreads = kThreadBlockSize;
  const std::size_t nVis = re.extent(0) * re.extent(1);
  const auto blocks = gridFor(nVis, kThreads);
  DYSCO_LAUNCH_1D(quantizeDitherKernel, blocks, kThreads, stream, re, im,
                  decValues, seedLo, seedHi, blockIndex, symbols);
}

}  // namespace gpu
}  // namespace dyscostman
