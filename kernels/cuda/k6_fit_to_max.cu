// K6 — "fit to maximum" post-normalization for the AF encoder.
//
// This file contains two implementations:
//
//   1. The default, fully grid-parallel *damped simultaneous* ("Jacobi")
//      algorithm described in `instructions-fit-to-max-gpu.md`.  It replaces
//      the old sequential greedy loop, which was the only sequential part of
//      the encoder and ran as a single 256-thread block (≈0.5 % of the GPU,
//      up to ~98 % of the total encode time).
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
//   * turns them into *damped* factors `G = cbrt(max(1, maxLevel/L))` and
//     `F = cbrt(max(1, maxLevel/M))` (only upscaling: the `max(1, ·)` guard),
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
// grid-stride pass over the data and reductions use `atomicMax` (order
// independent, therefore deterministic run-to-run).

#include "cuda_kernels.h"

#include <cmath>
#include <cstdlib>
#include <vector>

namespace dyscostman {
namespace gpu {

namespace {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

/// Component measure used by the greedy encoder: `max(|re|, |im|)`, written in
/// the same signed form as `AFTimeBlockEncoder::fitToMaximum` so that the
/// non-finite handling matches exactly.
__device__ __forceinline__ double comp(double r, double i) {
  return fmax(fmax(r, i), -fmin(r, i));
}

/// `atomicMax` for *non-negative* doubles.  The IEEE-754 bit pattern of a
/// non-negative double is order-isomorphic to an unsigned 64-bit integer, so
/// reinterpreting as `unsigned long long` makes the integer `atomicMax` compute
/// the double maximum.  `comp()` is always ≥ 0.  Being a maximum, the result is
/// independent of the order in which threads contribute (deterministic).
__device__ __forceinline__ void atomicMaxNonNeg(double *addr, double value) {
  atomicMax(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(__double_as_longlong(value)));
}

// ---------------------------------------------------------------------------
// Phase A — exact per-channel normalization to maxLevel
// ---------------------------------------------------------------------------

/// Reduces the per-visibility (channel×pol) maximum over cross-correlation
/// rows.  `L` must be zeroed beforehand.
__global__ void k6a_reduce_channel_max(const double *__restrict__ re,
                                       const double *__restrict__ im,
                                       const int *__restrict__ ant1,
                                       const int *__restrict__ ant2,
                                       std::size_t nRows,
                                       std::size_t visPerRow,
                                       double *__restrict__ L) {
  const std::size_t n = nRows * visPerRow;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    if (ant1[row] == ant2[row]) continue;  // skip autocorrelations
    const double c = comp(re[idx], im[idx]);
    if (!isfinite(c)) continue;
    atomicMaxNonNeg(&L[idx - row * visPerRow], c);
  }
}

/// Computes the phase-A factor `maxLevel / L[vi]` (no `max(1,·)` clamp: phase A
/// may downscale data that already exceeds maxLevel), divides the channel
/// metadata by it, and scales every element — including autocorrelations.
__global__ void k6a_apply_channel_fit(double *__restrict__ re,
                                      double *__restrict__ im,
                                      std::size_t n, std::size_t visPerRow,
                                      const double *__restrict__ L,
                                      float *__restrict__ meta,
                                      double maxLevel) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  // Metadata: only the first `visPerRow` global threads participate.
  for (std::size_t vi = start; vi < visPerRow; vi += stride) {
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / g0);
  }
  // Data: all threads, all rows.
  for (std::size_t i = start; i < n; i += stride) {
    const std::size_t vi = i % visPerRow;
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    re[i] *= g0;
    im[i] *= g0;
  }
}

// ---------------------------------------------------------------------------
// Phase B — damped simultaneous rounds
// ---------------------------------------------------------------------------

/// Per-round reduction: channel maxima `L` and antenna maxima `M` over
/// cross-correlation rows.  Both must be zeroed beforehand.
__global__ void k6b_reduce(const double *__restrict__ re,
                           const double *__restrict__ im,
                           const int *__restrict__ ant1,
                           const int *__restrict__ ant2, std::size_t nRows,
                           std::size_t nChannels, std::size_t nPol,
                           std::size_t nAnt, double *__restrict__ L,
                           double *__restrict__ M,
                           const int *__restrict__ doneFlag) {
  if (*doneFlag) return;  // converged: remaining rounds are no-ops
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t n = nRows * visPerRow;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    const int a1 = ant1[row];
    const int a2 = ant2[row];
    if (a1 == a2) continue;  // skip autocorrelations
    const std::size_t vi = idx - row * visPerRow;  // = ch*nPol + pol
    const std::size_t pol = vi % nPol;
    const double c = comp(re[idx], im[idx]);
    if (!isfinite(c)) continue;
    atomicMaxNonNeg(&L[vi], c);
    atomicMaxNonNeg(&M[pol * nAnt + static_cast<std::size_t>(a1)], c);
    atomicMaxNonNeg(&M[pol * nAnt + static_cast<std::size_t>(a2)], c);
  }
}

/// Per-round factor computation.  `L`/`M` are overwritten in place with the
/// damped factors `G`/`F`; the metadata is divided by them; `progressFlag` is
/// atomically raised while any axis still has meaningful headroom.
__global__ void k6b_factors(double *__restrict__ L, double *__restrict__ M,
                            std::size_t nChPol, std::size_t nPolAnt,
                            float *__restrict__ meta, std::size_t visPerRow,
                            double maxLevel, double epsChannel,
                            double epsAntenna, int *__restrict__ progressFlag,
                            const int *__restrict__ doneFlag) {
  if (*doneFlag) return;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int progress = 0;

  for (std::size_t vi = start; vi < nChPol; vi += stride) {
    const double l = L[vi];
    const double full =
        (maxLevel != 0.0 && l > 0.0) ? fmax(1.0, maxLevel / l) : 1.0;
    if (full > 1.0 + epsChannel) progress = 1;
    const double g = cbrt(full);
    L[vi] = g;  // now the factor, not the maximum
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / g);
  }
  for (std::size_t ai = start; ai < nPolAnt; ai += stride) {
    const double m = M[ai];
    const double full =
        (maxLevel != 0.0 && m > 0.0) ? fmax(1.0, maxLevel / m) : 1.0;
    if (full > 1.0 + epsAntenna) progress = 1;
    const double f = cbrt(full);
    M[ai] = f;
    meta[visPerRow + ai] =
        static_cast<float>(static_cast<double>(meta[visPerRow + ai]) / f);
  }

  if (progress) atomicOr(progressFlag, 1);
}

/// Per-round elementwise application of `G · F_{a1} · F_{a2}` to all elements.
__global__ void k6b_apply(double *__restrict__ re, double *__restrict__ im,
                          const int *__restrict__ ant1,
                          const int *__restrict__ ant2, std::size_t nRows,
                          std::size_t nChannels, std::size_t nPol,
                          std::size_t nAnt, const double *__restrict__ G,
                          const double *__restrict__ F,
                          const int *__restrict__ doneFlag) {
  if (*doneFlag) return;
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t n = nRows * visPerRow;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < n; idx += stride) {
    const std::size_t row = idx / visPerRow;
    const std::size_t vi = idx - row * visPerRow;
    const std::size_t pol = vi % nPol;
    // For autocorrelations (a1 == a2) the antenna factor enters twice, which
    // matches `AFTimeBlockEncoder::changeAntennaFactor`.
    const double scale =
        G[vi] *
        F[pol * nAnt + static_cast<std::size_t>(ant1[row])] *
        F[pol * nAnt + static_cast<std::size_t>(ant2[row])];
    re[idx] *= scale;
    im[idx] *= scale;
  }
}

/// Tiny end-of-round kernel: decides whether the next round is needed.  Kept
/// separate from `k6b_apply` so that no block of the apply kernel can observe
/// the new `doneFlag` and skip its share of the work.
__global__ void k6b_mark_done(int *__restrict__ doneFlag,
                              const int *__restrict__ progressFlag) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    *doneFlag = (*progressFlag == 0) ? 1 : 0;
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
__global__ void fitToMaxKernelGreedy(double *re, double *im, const int *ant1,
                                     const int *ant2, size_t nRows,
                                     size_t nChannels, size_t nPol, size_t nAnt,
                                     float *meta, double maxLevel,
                                     const int *csrRow,
                                     const size_t *csrOffset) {
  extern __shared__ double smem[];
  const size_t visPerRow = nChannels * nPol;
  double *factorCh = smem;                   // visPerRow
  double *largestCh = factorCh + visPerRow;  // nChannels
  double *increaseCh = largestCh + nChannels;  // nChannels
  double *maxComp = increaseCh + nChannels;    // nAnt
  double *increaseAnt = maxComp + nAnt;        // nAnt
  double *sel = increaseAnt + nAnt;  // 5 (doAntenna,index,factor,progressing,apply)
  const size_t tid = threadIdx.x;
  const size_t nth = blockDim.x;

  // ---- step A: scale every channel to the maximum encodable value ----
  for (size_t vi = tid; vi < visPerRow; vi += nth) {
    double best = 0.0;
    for (size_t row = 0; row != nRows; ++row) {
      if (ant1[row] != ant2[row]) {
        const size_t idx = row * visPerRow + vi;
        const double r = re[idx], v = im[idx];
        const double local = fmax(fmax(r, v), -fmin(r, v));
        if (isfinite(local) && local > best) best = local;
      }
    }
    factorCh[vi] = (maxLevel == 0.0 || best == 0.0) ? 1.0 : maxLevel / best;
  }
  __syncthreads();
  for (size_t vi = tid; vi < visPerRow; vi += nth)
    meta[vi] = static_cast<float>(static_cast<double>(meta[vi]) / factorCh[vi]);
  for (size_t idx = tid; idx < nRows * visPerRow; idx += nth) {
    const size_t vi = idx % visPerRow;
    re[idx] *= factorCh[vi];
    im[idx] *= factorCh[vi];
  }
  __syncthreads();

  for (size_t pol = 0; pol != nPol; ++pol) {
    int progressing = 1;
    while (progressing) {
      // channel candidates
      for (size_t c = tid; c < nChannels; c += nth) {
        const size_t vi = c * nPol + pol;
        double largest = 0.0;
        for (size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const size_t idx = row * visPerRow + vi;
            const double r = re[idx], v = im[idx];
            const double local = fmax(fmax(r, v), -fmin(r, v));
            if (isfinite(local) && local > largest) largest = local;
          }
        }
        largestCh[c] = largest;
        const double f = (largest == 0.0) ? 0.0 : (maxLevel / largest - 1.0);
        double increase = 0.0;
        for (size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const size_t idx = row * visPerRow + vi;
            const double av = fabs(re[idx] * f) + fabs(im[idx] * f);
            if (isfinite(av)) increase += av;
          }
        }
        increaseCh[c] = increase;
      }
      __syncthreads();
      // antenna candidates
      for (size_t a = tid; a < nAnt; a += nth) {
        double best = 0.0;
        for (size_t e = csrOffset[a]; e != csrOffset[a + 1]; ++e) {
          const size_t row = static_cast<size_t>(csrRow[e]);
          for (size_t channel = 0; channel != nChannels; ++channel) {
            const size_t idx = row * visPerRow + channel * nPol + pol;
            const double r = re[idx], v = im[idx];
            const double local = fmax(fmax(r, v), -fmin(r, v));
            if (isfinite(local) && local > best) best = local;
          }
        }
        maxComp[a] = best;
      }
      __syncthreads();
      for (size_t a = tid; a < nAnt; a += nth) {
        const double f =
            (maxComp[a] == 0.0) ? 0.0 : (maxLevel / maxComp[a] - 1.0);
        double acc = 0.0;
        for (size_t e = csrOffset[a]; e != csrOffset[a + 1]; ++e) {
          const size_t row = static_cast<size_t>(csrRow[e]);
          for (size_t channel = 0; channel != nChannels; ++channel) {
            const size_t idx = row * visPerRow + channel * nPol + pol;
            const double av = fabs(re[idx] * f) + fabs(im[idx] * f);
            if (isfinite(av)) acc += av;
          }
        }
        increaseAnt[a] = acc;
      }
      __syncthreads();

      // selection, strictly in the CPU order
      if (tid == 0) {
        double bestChannelIncrease = 0.0, channelFactor = 1.0;
        size_t bestChannel = 0;
        for (size_t c = 0; c != nChannels; ++c) {
          if (increaseCh[c] > bestChannelIncrease) {
            bestChannelIncrease = increaseCh[c];
            bestChannel = c;
            const double f = (largestCh[c] == 0.0)
                                 ? 0.0
                                 : (maxLevel / largestCh[c] - 1.0);
            channelFactor = f + 1.0;
          }
        }
        size_t bestAntenna = 0;
        double bestAntennaIncrease = 0.0;
        for (size_t a = 0; a != nAnt; ++a) {
          if (increaseAnt[a] > bestAntennaIncrease) {
            bestAntennaIncrease = increaseAnt[a];
            bestAntenna = a;
          }
        }
        int doAntenna, apply, prog;
        double factor;
        size_t index;
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
            const size_t mi = nChannels * nPol + nAnt * pol + index;
            meta[mi] = static_cast<float>(
                static_cast<double>(meta[mi]) / factor);
          } else {
            const size_t vi = index * nPol + pol;
            meta[vi] = static_cast<float>(
                static_cast<double>(meta[vi]) / factor);
          }
        }
      }
      __syncthreads();

      const int doAntenna = static_cast<int>(sel[0]);
      const size_t index = static_cast<size_t>(sel[1]);
      const double factor = sel[2];
      progressing = static_cast<int>(sel[3]);
      const int apply = static_cast<int>(sel[4]);
      if (apply) {
        if (doAntenna) {
          for (size_t row = tid; row < nRows; row += nth) {
            const size_t a1 = static_cast<size_t>(ant1[row]);
            const size_t a2 = static_cast<size_t>(ant2[row]);
            unsigned count = 0;
            if (a1 == index) ++count;
            if (a2 == index) ++count;
            for (unsigned repeat = 0; repeat != count; ++repeat)
              for (size_t ch = 0; ch != nChannels; ++ch) {
                const size_t idx = row * visPerRow + ch * nPol + pol;
                re[idx] *= factor;
                im[idx] *= factor;
              }
          }
        } else {
          const size_t vi = index * nPol + pol;
          for (size_t row = tid; row < nRows; row += nth) {
            const size_t idx = row * visPerRow + vi;
            re[idx] *= factor;
            im[idx] *= factor;
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

/// Number of blocks needed to cover `n` elements with `threads` threads, at
/// least one.  The kernels are grid-stride, so an over-estimate is harmless.
unsigned gridFor(std::size_t n, unsigned threads) {
  const std::size_t blocks = (n + threads - 1) / threads;
  return static_cast<unsigned>(blocks == 0 ? 1 : blocks);
}

/// RAII owner for one managed workspace allocation.
struct ManagedWorkspace {
  void *raw = nullptr;
  ~ManagedWorkspace() {
    if (raw != nullptr) (void)cudaFree(raw);
  }
};

/// Number of damped simultaneous rounds.  The early-exit flag usually stops
/// after 2–4; the remaining launches are no-ops.
constexpr unsigned kRounds = 5;

/// Damped simultaneous ("Jacobi") fit-to-max.  See the file header.
void launchJacobi(double *re, double *im, const int *ant1, const int *ant2,
                  std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                  std::size_t nAnt, float *meta, double maxLevel,
                  dysco_stream_t stream) {
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t n = nRows * visPerRow;
  const std::size_t lCount = visPerRow;
  const std::size_t mCount = nPol * nAnt;

  ManagedWorkspace ws;
  const std::size_t bytes = (lCount + mCount) * sizeof(double) +
                            2 * sizeof(int);
  DYSCO_MANAGED_ALLOC(&ws.raw, bytes);
  unsigned char *cursor = static_cast<unsigned char *>(ws.raw);
  double *L = reinterpret_cast<double *>(cursor);
  cursor += lCount * sizeof(double);
  double *M = reinterpret_cast<double *>(cursor);
  cursor += mCount * sizeof(double);
  int *progressFlag = reinterpret_cast<int *>(cursor);
  cursor += sizeof(int);
  int *doneFlag = reinterpret_cast<int *>(cursor);

  constexpr unsigned kBlock = 256;
  const unsigned grid = gridFor(n, kBlock);
  const unsigned factorsGrid = gridFor(lCount + mCount, kBlock);

  // ---- Phase A: exact per-channel normalization ----
  DYSCO_GPU_CHECK(
      cudaMemsetAsync(L, 0, lCount * sizeof(double), stream));
  k6a_reduce_channel_max<<<grid, kBlock, 0, stream>>>(
      re, im, ant1, ant2, nRows, visPerRow, L);
  k6a_apply_channel_fit<<<grid, kBlock, 0, stream>>>(re, im, n, visPerRow, L,
                                                     meta, maxLevel);

  // ---- Phase B: damped simultaneous rounds ----
  DYSCO_GPU_CHECK(cudaMemsetAsync(doneFlag, 0, sizeof(int), stream));
  for (unsigned round = 0; round != kRounds; ++round) {
    DYSCO_GPU_CHECK(cudaMemsetAsync(L, 0, lCount * sizeof(double), stream));
    DYSCO_GPU_CHECK(cudaMemsetAsync(M, 0, mCount * sizeof(double), stream));
    DYSCO_GPU_CHECK(cudaMemsetAsync(progressFlag, 0, sizeof(int), stream));

    k6b_reduce<<<grid, kBlock, 0, stream>>>(
        re, im, ant1, ant2, nRows, nChannels, nPol, nAnt, L, M, doneFlag);
    k6b_factors<<<factorsGrid, kBlock, 0, stream>>>(
        L, M, lCount, mCount, meta, visPerRow, maxLevel,
        /*epsChannel=*/1e-3, /*epsAntenna=*/1e-2, progressFlag, doneFlag);
    k6b_apply<<<grid, kBlock, 0, stream>>>(re, im, ant1, ant2, nRows,
                                           nChannels, nPol, nAnt, L, M,
                                           doneFlag);
    k6b_mark_done<<<1, 1, 0, stream>>>(doneFlag, progressFlag);
  }
  DYSCO_GPU_CHECK(cudaGetLastError());
}

/// Legacy greedy fit-to-max (bit-identical to the CPU encoder).
void launchGreedy(double *re, double *im, const int *ant1, const int *ant2,
                  std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                  std::size_t nAnt, float *meta, double maxLevel,
                  dysco_stream_t stream) {
  const std::size_t visPerRow = nChannels * nPol;

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
  std::vector<int> csrRow(total != 0 ? total : 1);
  std::vector<std::size_t> cursor(offset.begin(), offset.end() - 1);
  for (std::size_t row = 0; row != nRows; ++row) {
    const int a1 = ant1[row];
    const int a2 = ant2[row];
    if (a1 == a2) continue;
    csrRow[cursor[static_cast<std::size_t>(a1)]++] = static_cast<int>(row);
    csrRow[cursor[static_cast<std::size_t>(a2)]++] = static_cast<int>(row);
  }

  int *dCsrRow = nullptr;
  std::size_t *dCsrOffset = nullptr;
  DYSCO_MANAGED_ALLOC(&dCsrRow, (total != 0 ? total : 1) * sizeof(int));
  DYSCO_MANAGED_ALLOC(&dCsrOffset, (nAnt + 1) * sizeof(std::size_t));
  std::copy(csrRow.begin(), csrRow.end(), dCsrRow);
  std::copy(offset.begin(), offset.end(), dCsrOffset);

  const std::size_t smemDoubles =
      visPerRow + 2 * nChannels + 2 * nAnt + 5;  // + sel[5]
  const std::size_t smemBytes = smemDoubles * sizeof(double);
  constexpr unsigned kThreads = 256;
  cudaFuncSetAttribute(fitToMaxKernelGreedy,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smemBytes));
  fitToMaxKernelGreedy<<<1, kThreads, smemBytes, stream>>>(
      re, im, ant1, ant2, nRows, nChannels, nPol, nAnt, meta, maxLevel,
      dCsrRow, dCsrOffset);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  DYSCO_FREE(dCsrRow);
  DYSCO_FREE(dCsrOffset);
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry point (signature unchanged)
// ---------------------------------------------------------------------------

void k6_fit_to_max(dysco::span<double> re, dysco::span<double> im,
                   dysco::span<const int> ant1, dysco::span<const int> ant2,
                   std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                   std::size_t nAnt, dysco::span<float> meta, double maxLevel,
                   dysco_stream_t stream) {
  if (nRows == 0 || nChannels == 0 || nPol == 0 || nAnt == 0 ||
      maxLevel == 0.0)
    return;

  const char *greedy = std::getenv("DYSCO_K6_GREEDY");
  if (greedy != nullptr && greedy[0] != '\0' && greedy[0] != '0') {
    launchGreedy(re.data(), im.data(), ant1.data(), ant2.data(), nRows,
                 nChannels, nPol, nAnt, meta.data(), maxLevel, stream);
    return;
  }
  launchJacobi(re.data(), im.data(), ant1.data(), ant2.data(), nRows,
               nChannels, nPol, nAnt, meta.data(), maxLevel, stream);
}

}  // namespace gpu
}  // namespace dyscostman
