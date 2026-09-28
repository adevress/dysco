// K6 — greedy "fit to maximum" post-normalization.
//
// This is the only sequential part of the AF encoder.  It is implemented as a
// single persistent thread block: the whole greedy loop (candidate evaluation,
// selection and application) runs inside one kernel, so the intermediate
// arrays stay on the device and no per-iteration host round-trip is needed.
//
// The accumulation order per channel / per antenna is the same as the CPU
// (rows in order, columns in order), and the selection by thread 0 replicates
// the CPU's strict-greater tie-breaking, so the result is bit-identical to
// `AFTimeBlockEncoder::fitToMaximum`.
//
// Per-antenna reductions use a CSR list of the rows each antenna participates
// in, which removes an O(nAnt) factor compared to a per-antenna full scan.

#include "cuda_kernels.h"

#include <cmath>
#include <vector>

namespace dyscostman {
namespace gpu {

namespace {

__global__ void fitToMaxKernel(double *re, double *im, const int *ant1,
                               const int *ant2, size_t nRows, size_t nChannels,
                               size_t nPol, size_t nAnt, float *meta,
                               double maxLevel, const int *csrRow,
                               const size_t *csrOffset) {
  extern __shared__ double smem[];
  const size_t visPerRow = nChannels * nPol;
  double *factorCh = smem;                      // visPerRow
  double *largestCh = factorCh + visPerRow;     // nChannels
  double *increaseCh = largestCh + nChannels;   // nChannels
  double *maxComp = increaseCh + nChannels;     // nAnt
  double *increaseAnt = maxComp + nAnt;         // nAnt
  double *sel = increaseAnt + nAnt;             // 5 (doAntenna,index,factor,progressing,apply)
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

}  // namespace

void k6_fit_to_max(dysco::span<double> re, dysco::span<double> im,
                   dysco::span<const int> ant1, dysco::span<const int> ant2,
                   std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                   std::size_t nAnt, dysco::span<float> meta, double maxLevel,
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
  cudaFuncSetAttribute(fitToMaxKernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smemBytes));
  fitToMaxKernel<<<1, kThreads, smemBytes, stream>>>(
      re.data(), im.data(), ant1.data(), ant2.data(), nRows, nChannels, nPol,
      nAnt, meta.data(), maxLevel, dCsrRow, dCsrOffset);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  DYSCO_FREE(dCsrRow);
  DYSCO_FREE(dCsrOffset);
}

}  // namespace gpu
}  // namespace dyscostman
