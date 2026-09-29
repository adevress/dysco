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

namespace dyscostman {
namespace gpu {

namespace {

__global__ void initNaNKernel(cuda::std::span<double> matrix) {
  const std::size_t stride = gridStrideSize();
  for (std::size_t i = threadFlattenedId(); i < matrix.size(); i += stride) {
    matrix[i] = maxKeyToDouble(kQuietNaNBits);
  }
}

__global__ void antennaRmsMatrixKernel(MdSpan2D<const double> re,
                                       MdSpan2D<const double> im,
                                       cuda::std::span<const int> ant1,
                                       cuda::std::span<const int> ant2,
                                       std::size_t nChannels, std::size_t nPol,
                                       std::size_t pol,
                                       MdSpan2D<double> matrix) {
  const std::size_t row = threadFlattenedId();
  if (row >= re.extent(0)) return;
  const std::size_t nAnt = matrix.extent(0);
  std::size_t a1 = static_cast<std::size_t>(ant1[row]);
  std::size_t a2 = static_cast<std::size_t>(ant2[row]);
  if (a1 == a2) return;
  if (a1 > a2) {
    const std::size_t t = a1;
    a1 = a2;
    a2 = t;
  }
  double value = 0.0;
  std::size_t count = 0;
  for (std::size_t ch = 0; ch != nChannels; ++ch) {
    const std::size_t vi = ch * nPol + pol;
    const double r = re(row, vi);
    const double v = im(row, vi);
    if (cuda::std::isfinite(r) && cuda::std::isfinite(v)) {
      ++count;
      value += r * r + v * v;
    }
  }
  const double rms = cuda::std::sqrt(value / (count * 2.0));
  matrix(a1, a2) = rms;
  matrix(a2, a1) = rms;
}

}  // namespace

void k3_antenna_rms_matrix(MdSpan2D<const double> re,
                           MdSpan2D<const double> im,
                           cuda::std::span<const int> ant1,
                           cuda::std::span<const int> ant2,
                           std::size_t nChannels, std::size_t nPol,
                           std::size_t pol, MdSpan2D<double> matrix,
                           dysco_stream_t stream) {
  constexpr unsigned kThreads = kThreadBlockSize;
  const std::size_t matrixCount = matrix.extent(0) * matrix.extent(1);
  const auto initBlocks = gridFor(matrixCount, kThreads);
  DYSCO_LAUNCH_1D(initNaNKernel, initBlocks, kThreads, stream,
                  cuda::std::span<double>(matrix.data_handle(), matrixCount));
  const auto blocks = gridFor(re.extent(0), kThreads);
  DYSCO_LAUNCH_1D(antennaRmsMatrixKernel, blocks, kThreads, stream, re, im,
                  ant1, ant2, nChannels, nPol, pol, matrix);
}

}  // namespace gpu
}  // namespace dyscostman
