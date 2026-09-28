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
    re[i] /= rms;
    im[i] /= rms;
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

void k5a_apply_channel_norm(double *re, double *im, size_t nVis,
                            size_t visPerRow, const float *channelRms,
                            dysco_stream_t stream) {
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((nVis + threads - 1) / threads);
  DYSCO_LAUNCH_1D(applyChannelNormKernel, blocks, threads, stream, re, im, nVis,
                  visPerRow, channelRms);
}

void k5b_apply_antenna_norm(double *re, double *im, const int *ant1,
                            const int *ant2, size_t nRows, size_t nChannels,
                            size_t nPol, size_t pol,
                            const double *rmsPerAntenna, dysco_stream_t stream) {
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((nRows + threads - 1) / threads);
  DYSCO_LAUNCH_1D(applyAntennaNormKernel, blocks, threads, stream, re, im, ant1,
                  ant2, nRows, nChannels, nPol, pol, rmsPerAntenna);
}

}  // namespace gpu
}  // namespace dyscostman
