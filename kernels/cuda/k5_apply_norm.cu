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

__global__ void applyChannelNormKernel(double *re, double *im, size_t nVis,
                                       size_t visPerRow,
                                       const float *channelRms) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < nVis; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const double rms = static_cast<double>(channelRms[i % visPerRow]);
    // Match the CPU encoder: a zero RMS leaves the (zero) values untouched
    // instead of dividing 0/0 and producing NaNs.
    if (rms != 0.0) {
      re[i] /= rms;
      im[i] /= rms;
    }
  }
}

__global__ void applyAntennaNormKernel(double *re, double *im, const int *ant1,
                                       const int *ant2, size_t nRows,
                                       size_t nChannels, size_t nPol,
                                       size_t pol, const double *rmsPerAntenna) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= nRows) return;
  const double mul =
      rmsPerAntenna[static_cast<size_t>(ant1[row])] *
      rmsPerAntenna[static_cast<size_t>(ant2[row])];
  const double fac = (mul == 0.0) ? 0.0 : 1.0 / mul;
  const size_t base = row * nChannels * nPol;
  for (size_t ch = 0; ch != nChannels; ++ch) {
    const size_t idx = base + ch * nPol + pol;
    re[idx] *= fac;
    im[idx] *= fac;
  }
}

}  // namespace

void k5a_apply_channel_norm(dysco::span<double> re, dysco::span<double> im,
                            std::size_t visPerRow,
                            dysco::span<const float> channelRms,
                            dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t nVis = re.size();
  const auto blocks =
      static_cast<unsigned>((nVis + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(applyChannelNormKernel, blocks, kThreads, stream, re.data(),
                  im.data(), nVis, visPerRow, channelRms.data());
}

void k5b_apply_antenna_norm(dysco::span<double> re, dysco::span<double> im,
                            dysco::span<const int> ant1,
                            dysco::span<const int> ant2, std::size_t nRows,
                            std::size_t nChannels, std::size_t nPol,
                            std::size_t pol,
                            dysco::span<const double> rmsPerAntenna,
                            dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const auto blocks =
      static_cast<unsigned>((nRows + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(applyAntennaNormKernel, blocks, kThreads, stream, re.data(),
                  im.data(), ant1.data(), ant2.data(), nRows, nChannels, nPol,
                  pol, rmsPerAntenna.data());
}

}  // namespace gpu
}  // namespace dyscostman
