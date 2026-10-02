// K4 — iterative antenna-gain solver.
//
// Solves |V_ij| ~= g_i * g_j in a weighted, damped Jacobi fashion, mirroring
// the antenna-RMS solve in the CPU reference (kernels/af_reference.cc).  The
// whole solve runs in a single thread block; each thread reproduces the
// *sequential* inner loop over j, so the accumulation order is fixed and the
// result is deterministic (independent of scheduling).
//
// The two gain vectors live in the caller-provided buffers (`rms` and `next`)
// and each thread strides over the antennas, so a single kernel handles any
// antenna count.  Parallelizing the solve over several blocks would need a
// cooperative grid and is future work.
//
// The two block-wide reductions (the damped maximum and the convergence
// measure) use `cub::BlockReduce` + `cuda::maximum<>`: a maximum is
// order-independent, so the reduced value does not depend on the thread
// layout.  CUB defines that value in *thread 0 only*; the kernel broadcasts
// it through shared memory before any thread consumes it.  That is required
// for correctness, not just tidiness: `maxVal` decides which gains are zeroed
// and `precision` is the loop condition, so reading the undefined values
// could zero gains from garbage and let threads disagree on the number of
// loop iterations, deadlocking on the __syncthreads() barriers inside the
// loop.

#include "cuda_kernels.h"

#include <cub/cub.cuh>

#include <stdexcept>

namespace dyscostman {
namespace gpu {

namespace {

// Solver parameters, mirroring the CPU reference antenna-RMS solve
// (kernels/af_reference.cc).
constexpr std::uint32_t kAntennaSolveThreads = 1024;
constexpr std::uint32_t kMaxIterations = 100;
constexpr double kPrecisionTarget = 1e-6;
constexpr double kDampingNew = 0.8;      ///< weight of the new estimate
constexpr double kDampingOld = 1.0 - kDampingNew;
constexpr double kZeroThreshold = 1e-5;  ///< damped max below which a gain is zeroed

/// One damped Jacobi iteration of the antenna-gain solve.  A single block
/// handles all antennas: every antenna loop is thread-strided (so `nAnt` may
/// exceed the block size) and the gains live in the caller buffers `rms`
/// (current) and `next` (new estimate).
///
/// Per iteration: (1) `next[a] = sum_j matrix(a,j) / sum_j rms[j]` over the
/// finite off-diagonal entries with a non-zero gain, accumulated over j in
/// order; (2) damped update `rms[a] = next[a]*new + rms[a]*old`; (3)
/// block-wide max of `rms`, broadcast to every thread; (4) zero gains below
/// `maxVal * kZeroThreshold`, then block-wide max of the relative change,
/// which is the next loop condition.  `__syncthreads()` separates the phases
/// and the strided loops write disjoint antennas, so the result is
/// independent of thread ordering.  Threads without an antenna contribute
/// 0.0 to both maxima, the neutral element since gains and changes are >= 0.
__global__ void antennaGainSolveKernel(MdSpan2D<const double> matrix,
                                       cuda::std::span<double> rms,
                                       cuda::std::span<double> next) {
  using BlockReduce = cub::BlockReduce<double, kAntennaSolveThreads>;
  // Temp storage of the two reductions plus the slots through which thread 0
  // broadcasts their results (CUB defines them in thread 0 only) to the block.
  __shared__ typename BlockReduce::TempStorage reduceMax;
  __shared__ typename BlockReduce::TempStorage reducePrecision;
  __shared__ double broadcast[2];
  const std::size_t nAnt = matrix.extent(0);
  const std::size_t tid = threadIdx.x;
  const std::size_t stride = blockDim.x;

  for (std::size_t a = tid; a < nAnt; a += stride) rms[a] = 1.0;
  __syncthreads();

  double precision = 1.0;
  std::uint32_t iteration = 0;
  while (iteration < kMaxIterations && precision > kPrecisionTarget) {
    ++iteration;
    for (std::size_t a = tid; a < nAnt; a += stride) {
      double weightSum = 0.0;
      double acc = 0.0;
      for (std::size_t j = 0; j != nAnt; ++j) {
        if (a != j && rms[j] != 0.0) {
          const double m = matrix(a, j);
          if (cuda::std::isfinite(m)) {
            acc += m;
            weightSum += rms[j];
          }
        }
      }
      next[a] = (weightSum == 0.0) ? 0.0 : acc / weightSum;
    }
    __syncthreads();

    double localMax = 0.0;
    for (std::size_t a = tid; a < nAnt; a += stride) {
      rms[a] = next[a] * kDampingNew + rms[a] * kDampingOld;
      localMax = cuda::std::fmax(localMax, rms[a]);
    }
    const double reducedMax =
        BlockReduce(reduceMax).Reduce(localMax, cuda::maximum<>{});
    if (tid == 0) broadcast[0] = reducedMax;
    __syncthreads();
    const double maxVal = broadcast[0];

    double localPrecision = 0.0;
    for (std::size_t a = tid; a < nAnt; a += stride) {
      if (rms[a] < maxVal * kZeroThreshold) rms[a] = 0.0;
      const double p = cuda::std::fabs(rms[a] - next[a]) / maxVal;
      localPrecision = cuda::std::fmax(localPrecision, p);
    }
    const double reducedPrecision =
        BlockReduce(reducePrecision).Reduce(localPrecision, cuda::maximum<>{});
    if (tid == 0) broadcast[1] = reducedPrecision;
    __syncthreads();
    precision = broadcast[1];
  }
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
                           cuda::std::span<double> nextScratch,
                           dysco_stream_t stream) {
  const std::size_t nAnt = matrix.extent(0);
  if (nAnt == 0) return;
  if (nextScratch.size() < nAnt)
    throw std::runtime_error(
        "Dysco K4: antenna-solve scratch buffer is too small");
  antennaGainSolveKernel<<<1, kAntennaSolveThreads, 0, stream>>>(
      matrix, rmsPerAntenna, nextScratch);
  DYSCO_GPU_CHECK(cudaGetLastError());
}

void k4_store_antenna_gains(cuda::std::span<const double> rmsPerAntenna,
                            cuda::std::span<float> destination,
                            dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const std::size_t nAnt = rmsPerAntenna.size();
  const auto blocks = gridFor(nAnt, kThreads);
  DYSCO_LAUNCH_1D(storeGainsKernel, blocks, kThreads, stream, rmsPerAntenna,
                  destination);
}

}  // namespace gpu
}  // namespace dyscostman
