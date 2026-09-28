// K3 — per-baseline RMS matrix used by the antenna-gain solver.
//
// The CPU builds an nAnt x nAnt matrix where entry (a1,a2) is the RMS of all
// channels of the row with that antenna pair.  For a regular measurement set
// each unordered pair appears exactly once per time block, so this is a plain
// gather/reduction rather than a scatter: one work-item per row computes the
// RMS over the channels of its own row.  The matrix is pre-filled with NaN so
// that the solver treats absent baselines (including autocorrelations) as
// non-finite, exactly like the CPU code does for an empty RMSMeasurement.

#include "cuda_kernels.h"

#include <cmath>

namespace dyscostman {
namespace gpu {

namespace {

__global__ void initNaNKernel(double *matrix, size_t count) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    matrix[i] = nan("");
  }
}

__global__ void antennaRmsMatrixKernel(const double *re, const double *im,
                                       const int *ant1, const int *ant2,
                                       size_t nRows, size_t nChannels,
                                       size_t nPol, size_t nAnt, size_t pol,
                                       double *matrix) {
  const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= nRows) return;
  size_t a1 = static_cast<size_t>(ant1[row]);
  size_t a2 = static_cast<size_t>(ant2[row]);
  if (a1 == a2) return;
  if (a1 > a2) {
    const size_t t = a1;
    a1 = a2;
    a2 = t;
  }
  double value = 0.0;
  size_t count = 0;
  const size_t base = row * nChannels * nPol;
  for (size_t ch = 0; ch != nChannels; ++ch) {
    const size_t idx = base + ch * nPol + pol;
    const double r = re[idx];
    const double v = im[idx];
    if (isfinite(r) && isfinite(v)) {
      ++count;
      value += r * r + v * v;
    }
  }
  const double rms = sqrt(value / (count * 2.0));
  matrix[a1 * nAnt + a2] = rms;
  matrix[a2 * nAnt + a1] = rms;
}

}  // namespace

void k3_antenna_rms_matrix(const double *re, const double *im, const int *ant1,
                           const int *ant2, size_t nRows, size_t nChannels,
                           size_t nPol, size_t nAnt, size_t pol,
                           double *matrix, dysco_stream_t stream) {
  {
    const size_t count = nAnt * nAnt;
    const unsigned threads = 256;
    const unsigned blocks =
        static_cast<unsigned>((count + threads - 1) / threads);
    DYSCO_LAUNCH_1D(initNaNKernel, blocks, threads, stream, matrix, count);
  }
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((nRows + threads - 1) / threads);
  DYSCO_LAUNCH_1D(antennaRmsMatrixKernel, blocks, threads, stream, re, im,
                  ant1, ant2, nRows, nChannels, nPol, nAnt, pol, matrix);
}

}  // namespace gpu
}  // namespace dyscostman
