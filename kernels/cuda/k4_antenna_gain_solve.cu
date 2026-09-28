// K4 — iterative antenna-gain solver.
//
// Solves |V_ij| ~= g_i * g_j in a weighted, damped Jacobi fashion, mirroring
// `AFTimeBlockEncoder::calculateAntennaeRMS`.  The whole solve runs in a
// single thread block with one thread per antenna; each thread reproduces the
// *sequential* inner loop over j so the double-precision accumulation order is
// identical to the CPU version.  The damping/zeroing/precision step is
// performed by thread 0, again in the same order as the CPU code.

#include "cuda_kernels.h"

#include <cmath>

namespace dyscostman {
namespace gpu {

namespace {

__global__ void antennaGainSolveKernel(const double *matrix, size_t nAnt,
                                       double *rms) {
  const size_t i = threadIdx.x;
  extern __shared__ double smem[];
  double *next = smem;       // nAnt
  double *sharedPrec = smem + nAnt;  // 1

  if (i < nAnt) rms[i] = 1.0;
  __syncthreads();

  double precision = 1.0;
  int iteration = 0;
  while (iteration < 100 && precision > 1e-6) {
    ++iteration;
    if (i < nAnt) {
      double weightSum = 0.0;
      double acc = 0.0;
      for (size_t j = 0; j != nAnt; ++j) {
        if (i != j && rms[j] != 0.0) {
          const double w = rms[j];
          const double m = matrix[i * nAnt + j];
          if (isfinite(m)) {
            acc += m;
            weightSum += w;
          }
        }
      }
      next[i] = (weightSum == 0.0) ? 0.0 : acc / weightSum;
    }
    __syncthreads();

    if (i == 0) {
      double maxVal = 0.0;
      for (size_t k = 0; k != nAnt; ++k) {
        rms[k] = next[k] * 0.8 + rms[k] * 0.2;
        maxVal = fmax(maxVal, rms[k]);
      }
      double prec = 0.0;
      for (size_t k = 0; k != nAnt; ++k) {
        if (rms[k] < maxVal * 1e-5) rms[k] = 0.0;
        prec = fmax(prec, fabs(rms[k] - next[k]) / maxVal);
      }
      sharedPrec[0] = prec;
    }
    __syncthreads();
    precision = sharedPrec[0];
  }
}

__global__ void storeGainsKernel(const double *rms, size_t nAnt, float *dest) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nAnt) return;
  dest[i] = static_cast<float>(rms[i]);
}

}  // namespace

void k4_antenna_gain_solve(dysco::span<const double> matrix, std::size_t nAnt,
                           dysco::span<double> rmsPerAntenna,
                           dysco_stream_t stream) {
  const auto threads = static_cast<unsigned>(nAnt < 1024 ? nAnt : 1024);
  const std::size_t smemBytes = (nAnt + 1) * sizeof(double);
  antennaGainSolveKernel<<<1, threads, smemBytes, stream>>>(
      matrix.data(), nAnt, rmsPerAntenna.data());
}

void k4_store_antenna_gains(dysco::span<const double> rmsPerAntenna,
                            dysco::span<float> destination,
                            dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t nAnt = rmsPerAntenna.size();
  const auto blocks =
      static_cast<unsigned>((nAnt + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(storeGainsKernel, blocks, kThreads, stream,
                  rmsPerAntenna.data(), nAnt, destination.data());
}

}  // namespace gpu
}  // namespace dyscostman
