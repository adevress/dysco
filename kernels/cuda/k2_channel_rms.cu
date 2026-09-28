// K2 — per (channel, polarization) RMS normalization factor.
//
// Mirrors the CPU `RMSMeasurement` accumulation exactly: one work-item per
// (channel, polarization) plane walks the rows in order and accumulates
// re^2 + im^2 in double.  Keeping the row loop sequential is what makes the
// result bit-identical to the CPU encoder (a tree/parallel reduction would
// change the rounding of the sum).  With only nChannels*nPol work-items the
// occupancy is low, but the total amount of work is small compared to the
// quantization pass, so this is a deliberate trade of occupancy for
// reproducibility.

#include "cuda_kernels.h"

#include <cmath>

namespace dyscostman {
namespace gpu {

namespace {

__global__ void channelRmsKernel(const double *re, const double *im,
                                 size_t nRows, size_t visPerRow,
                                 float *channelRms) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= visPerRow) return;
  double value = 0.0;
  size_t count = 0;
  for (size_t row = 0; row != nRows; ++row) {
    const size_t idx = row * visPerRow + i;
    const double r = re[idx];
    const double v = im[idx];
    if (isfinite(r) && isfinite(v)) {
      ++count;
      value += r * r + v * v;
    }
  }
  channelRms[i] = static_cast<float>(sqrt(value / (count * 2.0)));
}

}  // namespace

void k2_channel_rms(dysco::span<const double> re, dysco::span<const double> im,
                    std::size_t nRows, std::size_t visPerRow,
                    dysco::span<float> channelRms, dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks =
      static_cast<unsigned>((visPerRow + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(channelRmsKernel, blocks, kThreads, stream, re.data(),
                  im.data(), nRows, visPerRow, channelRms.data());
}

}  // namespace gpu
}  // namespace dyscostman
