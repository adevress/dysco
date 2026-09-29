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

namespace dyscostman {
namespace gpu {

namespace {

__global__ void channelRmsKernel(MdSpan2D<const double> re,
                                 MdSpan2D<const double> im,
                                 cuda::std::span<float> channelRms) {
  const std::size_t visPerRow = re.extent(1);
  const std::size_t nRows = re.extent(0);
  const std::size_t i = threadFlattenedId();
  if (i >= visPerRow) return;
  double value = 0.0;
  std::size_t count = 0;
  for (std::size_t row = 0; row != nRows; ++row) {
    const double r = re(row, i);
    const double v = im(row, i);
    if (cuda::std::isfinite(r) && cuda::std::isfinite(v)) {
      ++count;
      value += r * r + v * v;
    }
  }
  channelRms[i] = static_cast<float>(cuda::std::sqrt(value / (count * 2.0)));
}

}  // namespace

void k2_channel_rms(MdSpan2D<const double> re, MdSpan2D<const double> im,
                    cuda::std::span<float> channelRms, dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const std::size_t visPerRow = re.extent(1);
  const auto blocks = gridFor(visPerRow, kThreads);
  DYSCO_LAUNCH_1D(channelRmsKernel, blocks, kThreads, stream, re, im,
                  channelRms);
}

}  // namespace gpu
}  // namespace dyscostman
