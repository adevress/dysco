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

void k7_quantize_dither(const double *re, const double *im, size_t nVis,
                        const float *decValues, size_t decCount, uint32_t seedLo,
                        uint32_t seedHi, uint64_t blockIndex,
                        unsigned int *symbols, dysco_stream_t stream) {
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((nVis + threads - 1) / threads);
  DYSCO_LAUNCH_1D(quantizeDitherKernel, blocks, threads, stream, re, im, nVis,
                  decValues, decCount, seedLo, seedHi, blockIndex, symbols);
}

}  // namespace gpu
}  // namespace dyscostman
