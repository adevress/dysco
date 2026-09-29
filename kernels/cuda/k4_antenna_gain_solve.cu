// K4 — iterative antenna-gain solver.
//
// Solves |V_ij| ~= g_i * g_j in a weighted, damped Jacobi fashion, mirroring
// `AFTimeBlockEncoder::calculateAntennaeRMS`.  The whole solve runs in a
// single thread block with one thread per antenna; each thread reproduces the
// *sequential* inner loop over j so the double-precision accumulation order is
// identical to the CPU version.
//
// The two block-wide reductions (the damped maximum and the convergence
// measure) are done with `cub::BlockReduce` instead of having thread 0 walk
// the arrays serially.  A maximum is order-independent, so the result is
// bit-for-bit the same as the sequential CPU version.  The working arrays live
// in shared memory, and the block size is fixed so it matches the
// compile-time `cub::BlockReduce` block size; idle threads contribute the
// neutral element to the reductions.

#include "cuda_kernels.h"

#include <cub/cub.cuh>

namespace dyscostman {
namespace gpu {

namespace {

constexpr unsigned kAntennaSolveThreads = 1024;
// Solver parameters, mirroring `AFTimeBlockEncoder::calculateAntennaeRMS`.
constexpr unsigned kMaxIterations = 100;
constexpr double kPrecisionTarget = 1e-6;
constexpr double kDampingNew = 0.8;      ///< weight of the new estimate
constexpr double kDampingOld = 1.0 - kDampingNew;
constexpr double kZeroThreshold = 1e-5;  ///< damped max below which a gain is zeroed
using BlockReduceDouble = cub::BlockReduce<double, kAntennaSolveThreads>;

struct GainSolveShared {
  double rms[kAntennaSolveThreads];
  double next[kAntennaSolveThreads];
  typename BlockReduceDouble::TempStorage reduceMax;
  typename BlockReduceDouble::TempStorage reducePrecision;
};

__global__ void antennaGainSolveKernel(MdSpan2D<const double> matrix,
                                       cuda::std::span<double> rmsOut) {
  __shared__ GainSolveShared shared;
  double *rms = shared.rms;
  double *next = shared.next;
  const std::size_t nAnt = matrix.extent(0);
  const std::size_t i = threadIdx.x;
  const bool active = i < nAnt;

  if (active) rms[i] = 1.0;
  __syncthreads();

  double precision = 1.0;
  unsigned iteration = 0;
  while (iteration < kMaxIterations && precision > kPrecisionTarget) {
    ++iteration;
    if (active) {
      double weightSum = 0.0;
      double acc = 0.0;
      for (std::size_t j = 0; j != nAnt; ++j) {
        if (i != j && rms[j] != 0.0) {
          const double m = matrix(i, j);
          if (cuda::std::isfinite(m)) {
            acc += m;
            weightSum += rms[j];
          }
        }
      }
      next[i] = (weightSum == 0.0) ? 0.0 : acc / weightSum;
    }
    __syncthreads();

    // Damped update (element-wise, so exactly reproducible) followed by the
    // order-independent maximum over all antennas.
    if (active) rms[i] = next[i] * kDampingNew + rms[i] * kDampingOld;
    const double maxVal =
        BlockReduceDouble(shared.reduceMax)
            .Reduce(active ? rms[i] : 0.0, cuda::maximum<>{});
    __syncthreads();

    if (active && rms[i] < maxVal * kZeroThreshold) rms[i] = 0.0;
    const double local =
        active ? cuda::std::fabs(rms[i] - next[i]) / maxVal : 0.0;
    precision = BlockReduceDouble(shared.reducePrecision)
                    .Reduce(local, cuda::maximum<>{});
    __syncthreads();
  }

  if (active) rmsOut[i] = rms[i];
}

__global__ void storeGainsKernel(cuda::std::span<const double> rms,
                                 cuda::std::span<float> destination) {
  const std::size_t i = threadFlattenedId();
  if (i >= rms.size()) return;
  destination[i] = static_cast<float>(rms[i]);
}

}  // namespace

void k4_antenna_gain_solve(MdSpan2D<const double> matrix,
                           cuda::std::span<double> rmsPerAntenna,
                           dysco_stream_t stream) {
  // The block size is fixed (CUB's BlockReduce block size is a compile-time
  // constant); threads without an antenna only contribute to the reductions.
  antennaGainSolveKernel<<<1, kAntennaSolveThreads, 0, stream>>>(
      matrix, rmsPerAntenna);
}

void k4_store_antenna_gains(cuda::std::span<const double> rmsPerAntenna,
                            cuda::std::span<float> destination,
                            dysco_stream_t stream) {
  constexpr unsigned kThreads = kThreadBlockSize;
  const std::size_t nAnt = rmsPerAntenna.size();
  const auto blocks = gridFor(nAnt, kThreads);
  DYSCO_LAUNCH_1D(storeGainsKernel, blocks, kThreads, stream, rmsPerAntenna,
                  destination);
}

}  // namespace gpu
}  // namespace dyscostman
