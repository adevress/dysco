// K6 — "fit to maximum" post-normalization for the AF encoder.
//
// This file contains two implementations:
//
//   1. The default, fully grid-parallel *damped simultaneous* ("Jacobi")
//      algorithm, which parallelizes the sequential greedy loop of
//      `AFTimeBlockEncoder::fitToMaximum` (the only sequential part of the
//      CPU encoder).
//
//   2. The legacy greedy kernel, kept for A/B benchmarking and selected with
//      the environment variable `DYSCO_K6_GREEDY=1`.  It is bit-identical to
//      `AFTimeBlockEncoder::fitToMaximum` (CPU greedy).
//
// The storage format is unchanged: only the resulting per-channel and
// per-antenna metadata factors are persisted, so the decoder does not care
// which algorithm produced them.
//
// ---------------------------------------------------------------------------
// Algorithm (Jacobi variant)
// ---------------------------------------------------------------------------
// After the RMS normalizations (K2..K5), the block is scaled so that its
// largest component approaches `maxLevel` (the largest decodable value of the
// quantizer).  The scale is factored into a per-(channel,pol) factor G and a
// per-(antenna,pol) factor F; an element `x` at row (a,b) becomes
//
//     x' = x · G[c,p] · F[a,p] · F[b,p]
//
// Phase A is the same exact per-channel normalization as before (it may
// downscale when the data exceeds `maxLevel`).  Phase B then runs a fixed
// number of rounds; each round:
//
//   * reduces the per-channel maximum L and the per-antenna maximum M over
//     cross-correlation rows only (autocorrelations are excluded, exactly as
//     in the greedy version);
//   * turns them into *damped* factors `G = cuda::std::cbrt(max(1, maxLevel/L))` and
//     `F = cuda::std::cbrt(max(1, maxLevel/M))` (only upscaling: the `max(1, ·)` guard),
//     updating the metadata by dividing by the factor;
//   * applies `G · F_a · F_b` to every element, including autocorrelations
//     (whose antenna factor therefore appears twice, i.e. `F²`).
//
// Because each element is scaled by exactly three factors, granting every axis
// the *cube root* of its full factor is always safe: `x ≤ min(L, M_a, M_b)`
// implies `x · (G·F_a·F_b) ≤ maxLevel`.  Hence no cross-correlation element
// can ever exceed `maxLevel`, and since every factor is ≥ 1 the total sum of
// absolute values is monotonically non-decreasing.  The log-headroom shrinks
// geometrically ((2/3)^R), so a handful of rounds is enough.  Every phase is a
// grid-stride pass over the data and reductions use atomic maxima (order
// independent, therefore deterministic run-to-run).  The per-round reductions
// aggregate into a block-local shared-memory histogram (`cuda::atomic_ref`
// with `thread_scope_block`, libcu++) and flush it to global memory once per
// block, and the reduction grid is sized through
// `cudaOccupancyMaxActiveBlocksPerMultiprocessor` so it stays fully resident.

#include "cuda_kernels.h"

#include <cstdlib>
#include <memory>
#include <vector>

namespace dyscostman {
namespace gpu {

namespace {

constexpr std::uint32_t kK6Block = 256;
constexpr std::uint32_t kMaxGridBlocks = 4096;

// Damping tolerances of the damped-simultaneous rounds: a channel factor is
// considered settled below a relative headroom of 0.1%, an antenna factor
// below 1% (mirrors the CPU greedy fit's stopping rules).
constexpr double kEpsChannel = 1e-3;
constexpr double kEpsAntenna = 1e-2;

// ---------------------------------------------------------------------------
// Phase A — exact per-channel normalization to maxLevel
// ---------------------------------------------------------------------------

/// Reduces the per-visibility (channel×pol) maximum over cross-correlation
/// rows.  `L` must be zeroed beforehand.  The block-local maxima are gathered
/// in shared memory before a single global atomic per entry per block.
__global__ void k6a_reduce_channel_max(MdSpan2D<const double> re,
                                       MdSpan2D<const double> im,
                                       cuda::std::span<const std::int32_t> ant1,
                                       cuda::std::span<const std::int32_t> ant2,
                                       cuda::std::span<double> L) {
  const std::size_t visPerRow = re.extent(1);
  extern __shared__ cuda::std::uint64_t sharedKeys[];
  for (std::size_t vi = threadIdx.x; vi < visPerRow; vi += kK6Block)
    sharedKeys[vi] = 0;
  __syncthreads();

  const std::size_t n = re.extent(0) * visPerRow;
  const std::size_t stride = gridStrideSize();
  for (std::size_t idx = threadFlattenedId(); idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    if (ant1[row] == ant2[row]) continue;  // skip autocorrelations
    const std::size_t vi = idx - row * visPerRow;
    const double c = componentMax(re(row, vi), im(row, vi));
    if (!cuda::std::isfinite(c)) continue;
    atomicMaxKeyBlock(&sharedKeys[vi], doubleToMaxKey(c));
  }
  __syncthreads();

  for (std::size_t vi = threadIdx.x; vi < visPerRow; vi += kK6Block)
    if (sharedKeys[vi] != 0)
      atomicMaxNonNegative(&L[vi], maxKeyToDouble(sharedKeys[vi]));
}

/// Computes the phase-A factor `maxLevel / L[vi]` (no `max(1,·)` clamp: phase A
/// may downscale data that already exceeds maxLevel), divides the channel
/// metadata by it, and scales every element — including autocorrelations.
__global__ void k6a_apply_channel_fit(MdSpan2D<double> re,
                                      MdSpan2D<double> im,
                                      cuda::std::span<const double> L,
                                      cuda::std::span<float> meta,
                                      double maxLevel) {
  const std::size_t visPerRow = re.extent(1);
  const std::size_t n = re.extent(0) * visPerRow;
  const std::size_t stride = gridStrideSize();
  const std::size_t start = threadFlattenedId();

  // Metadata: only the first `visPerRow` global threads participate.
  for (std::size_t vi = start; vi < visPerRow; vi += stride) {
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / g0);
  }
  // Data: all threads, all rows.
  for (std::size_t i = start; i < n; i += stride) {
    const std::size_t row = i / visPerRow;
    const std::size_t vi = i - row * visPerRow;
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    re(row, vi) *= g0;
    im(row, vi) *= g0;
  }
}

// ---------------------------------------------------------------------------
// Phase B — damped simultaneous rounds
// ---------------------------------------------------------------------------

/// Per-round reduction: channel maxima `L` and antenna maxima `M` over
/// cross-correlation rows.  Both must be zeroed beforehand.  Both are gathered
/// in one shared-memory histogram per block, then flushed with one global
/// atomic per entry per block.
__global__ void k6b_reduce(MdSpan2D<const double> re,
                           MdSpan2D<const double> im,
                           cuda::std::span<const std::int32_t> ant1,
                           cuda::std::span<const std::int32_t> ant2,
                           std::size_t nPol, std::size_t nAnt,
                           cuda::std::span<double> L,
                           cuda::std::span<double> M,
                           cuda::std::span<std::int32_t, 1> doneFlag) {
  if (doneFlag[0]) return;  // converged: remaining rounds are no-ops
  const std::size_t visPerRow = re.extent(1);
  const std::size_t nL = visPerRow;
  const std::size_t nM = nPol * nAnt;

  extern __shared__ cuda::std::uint64_t sharedKeys[];
  cuda::std::uint64_t *sL = sharedKeys;
  cuda::std::uint64_t *sM = sharedKeys + nL;
  for (std::size_t i = threadIdx.x; i < nL; i += kK6Block) sL[i] = 0;
  for (std::size_t i = threadIdx.x; i < nM; i += kK6Block) sM[i] = 0;
  __syncthreads();

  const std::size_t n = re.extent(0) * visPerRow;
  const std::size_t stride = gridStrideSize();
  for (std::size_t idx = threadFlattenedId(); idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    const std::int32_t a1 = ant1[row];
    const std::int32_t a2 = ant2[row];
    if (a1 == a2) continue;  // skip autocorrelations
    const std::size_t vi = idx - row * visPerRow;  // = ch*nPol + pol
    const std::size_t pol = vi % nPol;
    const double c = componentMax(re(row, vi), im(row, vi));
    if (!cuda::std::isfinite(c)) continue;
    const cuda::std::uint64_t key = doubleToMaxKey(c);
    atomicMaxKeyBlock(&sL[vi], key);
    atomicMaxKeyBlock(&sM[pol * nAnt + static_cast<std::size_t>(a1)], key);
    atomicMaxKeyBlock(&sM[pol * nAnt + static_cast<std::size_t>(a2)], key);
  }
  __syncthreads();

  for (std::size_t i = threadIdx.x; i < nL; i += kK6Block)
    if (sL[i] != 0)
      atomicMaxNonNegative(&L[i], maxKeyToDouble(sL[i]));
  for (std::size_t i = threadIdx.x; i < nM; i += kK6Block)
    if (sM[i] != 0)
      atomicMaxNonNegative(&M[i], maxKeyToDouble(sM[i]));
}

/// Per-round factor computation.  `L`/`M` are overwritten in place with the
/// damped factors `G`/`F`; the metadata is divided by them; `progressFlag` is
/// atomically raised while any axis still has meaningful headroom.
__global__ void k6b_factors(cuda::std::span<double> L,
                            cuda::std::span<double> M, std::size_t nChPol,
                            std::size_t nPolAnt, cuda::std::span<float> meta,
                            std::size_t visPerRow, double maxLevel,
                            cuda::std::span<std::int32_t, 1> progressFlag,
                            cuda::std::span<const std::int32_t, 1> doneFlag) {
  if (doneFlag[0]) return;
  const std::size_t stride = gridStrideSize();
  const std::size_t start = threadFlattenedId();
  std::int32_t progress = 0;

  for (std::size_t vi = start; vi < nChPol; vi += stride) {
    const double l = L[vi];
    const double full =
        (maxLevel != 0.0 && l > 0.0) ? cuda::std::fmax(1.0, maxLevel / l) : 1.0;
    if (full > 1.0 + kEpsChannel) progress = 1;
    const double g = cuda::std::cbrt(full);
    L[vi] = g;  // now the factor, not the maximum
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / g);
  }
  for (std::size_t ai = start; ai < nPolAnt; ai += stride) {
    const double m = M[ai];
    const double full =
        (maxLevel != 0.0 && m > 0.0) ? cuda::std::fmax(1.0, maxLevel / m) : 1.0;
    if (full > 1.0 + kEpsAntenna) progress = 1;
    const double f = cuda::std::cbrt(full);
    M[ai] = f;
    meta[visPerRow + ai] =
        static_cast<float>(static_cast<double>(meta[visPerRow + ai]) / f);
  }

  if (progress) atomicOrDevice(&progressFlag[0], 1);
}

/// Per-round elementwise application of `G · F_{a1} · F_{a2}` to all elements.
__global__ void k6b_apply(MdSpan2D<double> re, MdSpan2D<double> im,
                          cuda::std::span<const std::int32_t> ant1,
                          cuda::std::span<const std::int32_t> ant2, std::size_t nPol,
                          std::size_t nAnt, cuda::std::span<const double> G,
                          cuda::std::span<const double> F,
                          cuda::std::span<const std::int32_t, 1> doneFlag) {
  if (doneFlag[0]) return;
  const std::size_t visPerRow = re.extent(1);
  const std::size_t n = re.extent(0) * visPerRow;
  const std::size_t stride = gridStrideSize();
  for (std::size_t idx = threadFlattenedId(); idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    const std::size_t vi = idx - row * visPerRow;
    const std::size_t pol = vi % nPol;
    // For autocorrelations (a1 == a2) the antenna factor enters twice, which
    // matches `AFTimeBlockEncoder::changeAntennaFactor`.
    const double scale =
        G[vi] *
        F[pol * nAnt + static_cast<std::size_t>(ant1[row])] *
        F[pol * nAnt + static_cast<std::size_t>(ant2[row])];
    re(row, vi) *= scale;
    im(row, vi) *= scale;
  }
}

/// Tiny end-of-round kernel: decides whether the next round is needed.  Kept
/// separate from `k6b_apply` so that no block of the apply kernel can observe
/// the new `doneFlag` and skip its share of the work.
__global__ void k6b_mark_done(cuda::std::span<std::int32_t, 1> doneFlag,
                              cuda::std::span<const std::int32_t, 1> progressFlag) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    doneFlag[0] = (progressFlag[0] == 0) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Legacy greedy kernel (bit-identical to the CPU encoder)
// ---------------------------------------------------------------------------

// This is the only sequential part of the legacy AF encoder.  It is
// implemented as a single persistent thread block: the whole greedy loop
// (candidate evaluation, selection and application) runs inside one kernel, so
// the intermediate arrays stay on the device and no per-iteration host
// round-trip is needed.
//
// The accumulation order per channel / per antenna is the same as the CPU
// (rows in order, columns in order), and the selection by thread 0 replicates
// the CPU's strict-greater tie-breaking, so the result is bit-identical to
// `AFTimeBlockEncoder::fitToMaximum`.
//
// Per-antenna reductions use a CSR list of the rows each antenna participates
// in, which removes an O(nAnt) factor compared to a per-antenna full scan.
__global__ void fitToMaxKernelGreedy(
    MdSpan2D<double> re, MdSpan2D<double> im,
    cuda::std::span<const std::int32_t> ant1, cuda::std::span<const std::int32_t> ant2,
    std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
    cuda::std::span<float> meta, double maxLevel,
    cuda::std::span<const std::int32_t> csrRow,
    cuda::std::span<const std::size_t> csrOffset) {
  extern __shared__ double smem[];
  const std::size_t visPerRow = re.extent(1);
  const std::size_t nRows = re.extent(0);
  double *factorCh = smem;                   // visPerRow
  double *largestCh = factorCh + visPerRow;  // nChannels
  double *increaseCh = largestCh + nChannels;  // nChannels
  double *maxComp = increaseCh + nChannels;    // nAnt
  double *increaseAnt = maxComp + nAnt;        // nAnt
  double *sel = increaseAnt + nAnt;  // 5 (doAntenna,index,factor,progressing,apply)
  const std::size_t tid = threadIdx.x;
  const std::size_t nth = blockDim.x;

  // ---- step A: scale every channel to the maximum encodable value ----
  for (std::size_t vi = tid; vi < visPerRow; vi += nth) {
    double best = 0.0;
    for (std::size_t row = 0; row != nRows; ++row) {
      if (ant1[row] != ant2[row]) {
        const double local = componentMax(re(row, vi), im(row, vi));
        if (cuda::std::isfinite(local) && local > best) best = local;
      }
    }
    factorCh[vi] = (maxLevel == 0.0 || best == 0.0) ? 1.0 : maxLevel / best;
  }
  __syncthreads();
  for (std::size_t vi = tid; vi < visPerRow; vi += nth)
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / factorCh[vi]);
  for (std::size_t row = 0; row != nRows; ++row)
    for (std::size_t vi = tid; vi < visPerRow; vi += nth) {
      re(row, vi) *= factorCh[vi];
      im(row, vi) *= factorCh[vi];
    }
  __syncthreads();

  for (std::size_t pol = 0; pol != nPol; ++pol) {
    std::int32_t progressing = 1;
    while (progressing) {
      // channel candidates
      for (std::size_t c = tid; c < nChannels; c += nth) {
        const std::size_t vi = c * nPol + pol;
        double largest = 0.0;
        for (std::size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const double local = componentMax(re(row, vi), im(row, vi));
            if (cuda::std::isfinite(local) && local > largest) largest = local;
          }
        }
        largestCh[c] = largest;
        const double f = (largest == 0.0) ? 0.0 : (maxLevel / largest - 1.0);
        double increase = 0.0;
        for (std::size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const double av = cuda::std::fabs(re(row, vi) * f) + cuda::std::fabs(im(row, vi) * f);
            if (cuda::std::isfinite(av)) increase += av;
          }
        }
        increaseCh[c] = increase;
      }
      __syncthreads();
      // antenna candidates
      for (std::size_t a = tid; a < nAnt; a += nth) {
        double best = 0.0;
        for (std::size_t e = csrOffset[a]; e != csrOffset[a + 1]; ++e) {
          const std::size_t row = static_cast<std::size_t>(csrRow[e]);
          for (std::size_t channel = 0; channel != nChannels; ++channel) {
            const std::size_t vi = channel * nPol + pol;
            const double local = componentMax(re(row, vi), im(row, vi));
            if (cuda::std::isfinite(local) && local > best) best = local;
          }
        }
        maxComp[a] = best;
      }
      __syncthreads();
      for (std::size_t a = tid; a < nAnt; a += nth) {
        const double f =
            (maxComp[a] == 0.0) ? 0.0 : (maxLevel / maxComp[a] - 1.0);
        double acc = 0.0;
        for (std::size_t e = csrOffset[a]; e != csrOffset[a + 1]; ++e) {
          const std::size_t row = static_cast<std::size_t>(csrRow[e]);
          for (std::size_t channel = 0; channel != nChannels; ++channel) {
            const std::size_t vi = channel * nPol + pol;
            const double av = cuda::std::fabs(re(row, vi) * f) + cuda::std::fabs(im(row, vi) * f);
            if (cuda::std::isfinite(av)) acc += av;
          }
        }
        increaseAnt[a] = acc;
      }
      __syncthreads();

      // selection, strictly in the CPU order
      if (tid == 0) {
        double bestChannelIncrease = 0.0, channelFactor = 1.0;
        std::size_t bestChannel = 0;
        for (std::size_t c = 0; c != nChannels; ++c) {
          if (increaseCh[c] > bestChannelIncrease) {
            bestChannelIncrease = increaseCh[c];
            bestChannel = c;
            const double f = (largestCh[c] == 0.0)
                                 ? 0.0
                                 : (maxLevel / largestCh[c] - 1.0);
            channelFactor = f + 1.0;
          }
        }
        std::size_t bestAntenna = 0;
        double bestAntennaIncrease = 0.0;
        for (std::size_t a = 0; a != nAnt; ++a) {
          if (increaseAnt[a] > bestAntennaIncrease) {
            bestAntennaIncrease = increaseAnt[a];
            bestAntenna = a;
          }
        }
        std::int32_t doAntenna, apply, prog;
        double factor;
        std::size_t index;
        if (bestAntennaIncrease > bestChannelIncrease) {
          factor = (maxComp[bestAntenna] == 0.0)
                       ? 1.0
                       : (maxLevel / maxComp[bestAntenna]);
          doAntenna = 1;
          index = bestAntenna;
        } else {
          factor = channelFactor;
          doAntenna = 0;
          index = bestChannel;
        }
        apply = (factor >= 1.0) ? 1 : 0;
        if (doAntenna)
          prog = apply && (factor > 1.01);
        else
          prog = apply && (factor > 1.001);
        sel[0] = static_cast<double>(doAntenna);
        sel[1] = static_cast<double>(index);
        sel[2] = factor;
        sel[3] = static_cast<double>(prog);
        sel[4] = static_cast<double>(apply);
        // metadata is only updated when the factor is applied
        if (apply) {
          if (doAntenna) {
            const std::size_t mi = nChannels * nPol + nAnt * pol + index;
            meta[mi] =
                static_cast<float>(static_cast<double>(meta[mi]) / factor);
          } else {
            const std::size_t vi = index * nPol + pol;
            meta[vi] =
                static_cast<float>(static_cast<double>(meta[vi]) / factor);
          }
        }
      }
      __syncthreads();

      const std::int32_t doAntenna = static_cast<std::int32_t>(sel[0]);
      const std::size_t index = static_cast<std::size_t>(sel[1]);
      const double factor = sel[2];
      progressing = static_cast<std::int32_t>(sel[3]);
      const std::int32_t apply = static_cast<std::int32_t>(sel[4]);
      if (apply) {
        if (doAntenna) {
          for (std::size_t row = tid; row < nRows; row += nth) {
            const std::size_t a1 = static_cast<std::size_t>(ant1[row]);
            const std::size_t a2 = static_cast<std::size_t>(ant2[row]);
            std::uint32_t count = 0;
            if (a1 == index) ++count;
            if (a2 == index) ++count;
            for (std::uint32_t repeat = 0; repeat != count; ++repeat)
              for (std::size_t ch = 0; ch != nChannels; ++ch) {
                const std::size_t vi = ch * nPol + pol;
                re(row, vi) *= factor;
                im(row, vi) *= factor;
              }
          }
        } else {
          const std::size_t vi = index * nPol + pol;
          for (std::size_t row = tid; row < nRows; row += nth) {
            re(row, vi) *= factor;
            im(row, vi) *= factor;
          }
        }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

/// Size a grid-stride launch to the number of blocks that actually fit on the
/// device (occupancy API), capped by the amount of work.  This keeps the
/// shared-memory reduction grid resident and avoids over-subscription.
template <class Kernel>
std::uint32_t residentGrid(Kernel kernel, std::size_t n,
                           std::uint32_t threads, std::size_t sharedBytes) {
  const std::size_t needed = (n + threads - 1) / threads;
  std::int32_t device = 0;
  std::int32_t smCount = 0;
  std::int32_t blocksPerSm = 0;
  if (cudaGetDevice(&device) == cudaSuccess &&
      cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount,
                             device) == cudaSuccess &&
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel,
                                                    threads,
                                                    sharedBytes) == cudaSuccess &&
      blocksPerSm > 0) {
    const std::size_t capacity = static_cast<std::size_t>(smCount) *
                                 static_cast<std::size_t>(blocksPerSm);
    const std::size_t blocks = needed == 0 ? 1 : needed;
    const std::size_t capped = blocks < capacity ? blocks : capacity;
    return static_cast<std::uint32_t>(
        capped < kMaxGridBlocks ? capped : kMaxGridBlocks);
  }
  return gridFor(n, threads);
}

/// Number of damped simultaneous rounds.  The early-exit flag usually stops
/// after 2–4; the remaining launches are no-ops.
constexpr std::uint32_t kRounds = 5;

/// Damped simultaneous ("Jacobi") fit-to-max.  See the file header.
void launchJacobi(MdSpan2D<double> re, MdSpan2D<double> im,
                  cuda::std::span<const std::int32_t> ant1,
                  cuda::std::span<const std::int32_t> ant2, std::size_t nChannels,
                  std::size_t nPol, std::size_t nAnt,
                  cuda::std::span<float> meta, double maxLevel,
                  dysco_stream_t stream) {
  const std::size_t visPerRow = re.extent(1);
  const std::size_t n = re.extent(0) * visPerRow;
  const std::size_t lCount = visPerRow;
  const std::size_t mCount = nPol * nAnt;

  // Managed workspace: [L, M, progressFlag, doneFlag], owned as RAII.
  const std::size_t bytes =
      (lCount + mCount) * sizeof(double) + 2 * sizeof(std::int32_t);
  void *raw = nullptr;
  DYSCO_MANAGED_ALLOC(&raw, bytes);
  const std::unique_ptr<void, dysco_gpu::ManagedDeleter> workspace(raw);
  std::uint8_t *cursor = static_cast<std::uint8_t *>(raw);
  double *L = reinterpret_cast<double *>(cursor);
  cursor += lCount * sizeof(double);
  double *M = reinterpret_cast<double *>(cursor);
  cursor += mCount * sizeof(double);
  std::int32_t *progressFlag = reinterpret_cast<std::int32_t *>(cursor);
  cursor += sizeof(std::int32_t);
  std::int32_t *doneFlag = reinterpret_cast<std::int32_t *>(cursor);

  cuda::std::span<double> lView(L, lCount);
  cuda::std::span<double> mView(M, mCount);
  cuda::std::span<std::int32_t, 1> progressView(progressFlag, 1);
  cuda::std::span<std::int32_t, 1> doneView(doneFlag, 1);

  const std::size_t reduceSmem =
      (lCount + mCount) * sizeof(cuda::std::uint64_t);
  const std::size_t phaseASmem = lCount * sizeof(cuda::std::uint64_t);
  const std::uint32_t applyGrid = gridFor(n, kK6Block);
  const std::uint32_t reduceGrid =
      residentGrid(k6b_reduce, n, kK6Block, reduceSmem);

  // ---- Phase A: exact per-channel normalization ----
  DYSCO_GPU_CHECK(cudaMemsetAsync(L, 0, lCount * sizeof(double), stream));
  k6a_reduce_channel_max<<<reduceGrid, kK6Block, phaseASmem, stream>>>(
      re, im, ant1, ant2, lView);
  k6a_apply_channel_fit<<<applyGrid, kK6Block, 0, stream>>>(re, im, lView,
                                                            meta, maxLevel);

  // ---- Phase B: damped simultaneous rounds ----
  DYSCO_GPU_CHECK(
      cudaMemsetAsync(doneFlag, 0, sizeof(std::int32_t), stream));
  for (std::uint32_t round = 0; round != kRounds; ++round) {
    DYSCO_GPU_CHECK(cudaMemsetAsync(L, 0, lCount * sizeof(double), stream));
    DYSCO_GPU_CHECK(cudaMemsetAsync(M, 0, mCount * sizeof(double), stream));
    DYSCO_GPU_CHECK(
        cudaMemsetAsync(progressFlag, 0, sizeof(std::int32_t), stream));

    k6b_reduce<<<reduceGrid, kK6Block, reduceSmem, stream>>>(
        re, im, ant1, ant2, nPol, nAnt, lView, mView, doneView);
    k6b_factors<<<gridFor(lCount + mCount, kK6Block), kK6Block, 0, stream>>>(
        lView, mView, lCount, mCount, meta, visPerRow, maxLevel, progressView,
        doneView);
    k6b_apply<<<applyGrid, kK6Block, 0, stream>>>(re, im, ant1, ant2, nPol,
                                                  nAnt, lView, mView, doneView);
    k6b_mark_done<<<1, 1, 0, stream>>>(doneView, progressView);
  }
  DYSCO_GPU_CHECK(cudaGetLastError());
}

/// Legacy greedy fit-to-max (bit-identical to the CPU encoder).
void launchGreedy(MdSpan2D<double> re, MdSpan2D<double> im,
                  cuda::std::span<const std::int32_t> ant1,
                  cuda::std::span<const std::int32_t> ant2, std::size_t nChannels,
                  std::size_t nPol, std::size_t nAnt,
                  cuda::std::span<float> meta, double maxLevel,
                  dysco_stream_t stream) {
  const std::size_t visPerRow = re.extent(1);
  const std::size_t nRows = re.extent(0);

  // CSR: rows each antenna participates in (each non-autocorr row -> 2).
  std::vector<std::size_t> offset(nAnt + 1, 0);
  for (std::size_t row = 0; row != nRows; ++row) {
    const auto a1 = static_cast<std::size_t>(ant1[row]);
    const auto a2 = static_cast<std::size_t>(ant2[row]);
    if (a1 == a2) continue;
    ++offset[a1 + 1];
    ++offset[a2 + 1];
  }
  for (std::size_t a = 0; a != nAnt; ++a) offset[a + 1] += offset[a];
  const std::size_t total = offset[nAnt];
  std::vector<std::int32_t> csrRow(total != 0 ? total : 1);
  std::vector<std::size_t> cursor(offset.begin(), offset.end() - 1);
  for (std::size_t row = 0; row != nRows; ++row) {
    const std::int32_t a1 = ant1[row];
    const std::int32_t a2 = ant2[row];
    if (a1 == a2) continue;
    csrRow[cursor[static_cast<std::size_t>(a1)]++] =
        static_cast<std::int32_t>(row);
    csrRow[cursor[static_cast<std::size_t>(a2)]++] =
        static_cast<std::int32_t>(row);
  }

  std::int32_t *dCsrRow = nullptr;
  std::size_t *dCsrOffset = nullptr;
  DYSCO_MANAGED_ALLOC(&dCsrRow, (total != 0 ? total : 1) * sizeof(std::int32_t));
  DYSCO_MANAGED_ALLOC(&dCsrOffset, (nAnt + 1) * sizeof(std::size_t));
  std::copy(csrRow.begin(), csrRow.end(), dCsrRow);
  std::copy(offset.begin(), offset.end(), dCsrOffset);

  const std::size_t smemDoubles =
      visPerRow + 2 * nChannels + 2 * nAnt + 5;  // + sel[5]
  const std::size_t smemBytes = smemDoubles * sizeof(double);
  const std::uint32_t kThreads = kK6Block;
  cudaFuncSetAttribute(fitToMaxKernelGreedy,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<std::int32_t>(smemBytes));
  fitToMaxKernelGreedy<<<1, kThreads, smemBytes, stream>>>(
      re, im, ant1, ant2, nChannels, nPol, nAnt, meta, maxLevel,
      cuda::std::span<const std::int32_t>(dCsrRow, total != 0 ? total : 1),
      cuda::std::span<const std::size_t>(dCsrOffset, nAnt + 1));
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  DYSCO_FREE(dCsrRow);
  DYSCO_FREE(dCsrOffset);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void k6_fit_to_max(MdSpan2D<double> re, MdSpan2D<double> im,
                   cuda::std::span<const std::int32_t> ant1,
                   cuda::std::span<const std::int32_t> ant2, std::size_t nChannels,
                   std::size_t nPol, std::size_t nAnt,
                   cuda::std::span<float> meta, double maxLevel,
                   dysco_stream_t stream) {
  if (re.extent(0) == 0 || nChannels == 0 || nPol == 0 || nAnt == 0 ||
      maxLevel == 0.0)
    return;

  const char *greedy = std::getenv("DYSCO_K6_GREEDY");
  if (greedy != nullptr && greedy[0] != '\0' && greedy[0] != '0') {
    launchGreedy(re, im, ant1, ant2, nChannels, nPol, nAnt, meta, maxLevel,
                 stream);
    return;
  }
  launchJacobi(re, im, ant1, ant2, nChannels, nPol, nAnt, meta, maxLevel,
               stream);
}

}  // namespace gpu
}  // namespace dyscostman
