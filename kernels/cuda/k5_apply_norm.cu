// K5 — apply the normalization factors.
//
// K5a divides every value by its channel RMS (after K2 has computed all of
// them, exactly like the CPU does in a second pass), K5b multiplies by
// g_a1 * g_a2 for the polarization currently being processed.  The operation
// order (divide first, then multiply) is kept identical to the CPU encoder so
// that the double results match bit for bit.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void applyChannelNormKernel(cuda::std::span<double> re,
                                       cuda::std::span<double> im,
                                       std::size_t visPerRow,
                                       cuda::std::span<const float> channelRms) {
  const std::size_t nVis = re.size();
  const std::size_t stride = gridStrideSize();
  for (std::size_t i = threadFlattenedId(); i < nVis; i += stride) {
    const double rms = static_cast<double>(channelRms[i % visPerRow]);
    // Match the CPU encoder: a zero RMS leaves the (zero) values untouched
    // instead of dividing 0/0 and producing NaNs.
    if (rms != 0.0) {
      re[i] /= rms;
      im[i] /= rms;
    }
  }
}

__global__ void applyAntennaNormKernel(MdSpan2D<double> re,
                                       MdSpan2D<double> im,
                                       cuda::std::span<const std::int32_t> ant1,
                                       cuda::std::span<const std::int32_t> ant2,
                                       std::size_t nChannels, std::size_t nPol,
                                       std::size_t pol,
                                       cuda::std::span<const double> rmsPerAntenna) {
  const std::size_t row = threadFlattenedId();
  if (row >= re.extent(0)) return;
  const double mul =
      rmsPerAntenna[static_cast<std::size_t>(ant1[row])] *
      rmsPerAntenna[static_cast<std::size_t>(ant2[row])];
  const double fac = (mul == 0.0) ? 0.0 : 1.0 / mul;
  for (std::size_t ch = 0; ch != nChannels; ++ch) {
    const std::size_t vi = ch * nPol + pol;
    re(row, vi) *= fac;
    im(row, vi) *= fac;
  }
}

}  // namespace

void k5a_apply_channel_norm(cuda::std::span<double> re,
                            cuda::std::span<double> im, std::size_t visPerRow,
                            cuda::std::span<const float> channelRms,
                            dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const std::size_t nVis = re.size();
  const auto blocks = gridFor(nVis, kThreads);
  DYSCO_LAUNCH_1D(applyChannelNormKernel, blocks, kThreads, stream, re, im,
                  visPerRow, channelRms);
}

void k5b_apply_antenna_norm(MdSpan2D<double> re, MdSpan2D<double> im,
                            cuda::std::span<const std::int32_t> ant1,
                            cuda::std::span<const std::int32_t> ant2,
                            std::size_t nChannels, std::size_t nPol,
                            std::size_t pol,
                            cuda::std::span<const double> rmsPerAntenna,
                            dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const auto blocks = gridFor(re.extent(0), kThreads);
  DYSCO_LAUNCH_1D(applyAntennaNormKernel, blocks, kThreads, stream, re, im,
                  ant1, ant2, nChannels, nPol, pol, rmsPerAntenna);
}

}  // namespace gpu
}  // namespace dyscostman
