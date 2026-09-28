// K7 — dithered dictionary quantization.
//
// Reproduces `StochasticEncoder<float>::EncodeWithDithering` for every value
// of the block.  The binary search is performed over the *decode* values (the
// centroids), exactly like the CPU encoder.  The dither value comes from the
// counter-based Philox generator, indexed by the global symbol index, so the
// sequence is independent of how the work is distributed over threads.

#include "cuda_kernels.h"
#include "philox_rng.h"

#include <cmath>

namespace dyscostman {
namespace gpu {

namespace {

DYSCO_HD DYSCO_INLINE unsigned int quantizeDither(float value,
                                                  const float *decValues,
                                                  size_t decCount,
                                                  unsigned int dither) {
  if (!isfinite(value)) return static_cast<unsigned int>(decCount);
  // lower_bound over decValues: first index with decValues[index] >= value.
  size_t lo = 0, hi = decCount;
  while (lo < hi) {
    const size_t mid = (lo + hi) >> 1;
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
      static_cast<float>(2147483648.0f * (value - leftValue) /
                         (rightValue - leftValue));
  if (ditherMark > static_cast<float>(dither))
    return static_cast<unsigned int>(lo);
  else
    return static_cast<unsigned int>(lo - 1);
}

// Flat variant used by the tests: one symbol per input value, with the dither
// indexed by the value index (the AF path uses 2*i / 2*i+1).
__global__ void quantizeFlatKernel(const float *values, size_t count,
                                   const float *decValues, size_t decCount,
                                   uint32_t seedLo, uint32_t seedHi,
                                   uint64_t blockIndex,
                                   unsigned int *symbols) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const unsigned int dither =
      random123::ditherValue(seedLo, seedHi, blockIndex, i);
  symbols[i] = quantizeDither(values[i], decValues, decCount, dither);
}

__global__ void quantizeDitherKernel(const double *re, const double *im,
                                     size_t nVis, const float *decValues,
                                     size_t decCount, uint32_t seedLo,
                                     uint32_t seedHi, uint64_t blockIndex,
                                     unsigned int *symbols) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nVis) return;
  const uint64_t base = static_cast<uint64_t>(i) * 2;
  const unsigned int dRe = random123::ditherValue(
      seedLo, seedHi, blockIndex, base);
  const unsigned int dIm = random123::ditherValue(
      seedLo, seedHi, blockIndex, base + 1);
  symbols[base] = quantizeDither(static_cast<float>(re[i]), decValues, decCount,
                                 dRe);
  symbols[base + 1] = quantizeDither(static_cast<float>(im[i]), decValues,
                                     decCount, dIm);
}

}  // namespace

void k7_quantize_flat(dysco::span<const float> values,
                      dysco::span<const float> decValues, std::uint32_t seedLo,
                      std::uint32_t seedHi, std::uint64_t blockIndex,
                      dysco::span<unsigned int> symbols,
                      dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t count = values.size();
  const auto blocks =
      static_cast<unsigned>((count + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(quantizeFlatKernel, blocks, kThreads, stream, values.data(),
                  count, decValues.data(), decValues.size(), seedLo, seedHi,
                  blockIndex, symbols.data());
}

void k7_quantize_dither(dysco::span<const double> re,
                        dysco::span<const double> im,
                        dysco::span<const float> decValues,
                        std::uint32_t seedLo, std::uint32_t seedHi,
                        std::uint64_t blockIndex,
                        dysco::span<unsigned int> symbols,
                        dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t nVis = re.size();
  const std::size_t decCount = decValues.size();
  const auto blocks =
      static_cast<unsigned>((nVis + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(quantizeDitherKernel, blocks, kThreads, stream, re.data(),
                  im.data(), nVis, decValues.data(), decCount, seedLo, seedHi,
                  blockIndex, symbols.data());
}

}  // namespace gpu
}  // namespace dyscostman
