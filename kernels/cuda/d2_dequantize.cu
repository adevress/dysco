// D2 — dequantization / decoding.
//
// value = decValues[symbol] * channelRms[ch,pol] * g_a1 * g_a2
// with the NaN symbol (index == decCount) producing a NaN visibility, exactly
// like the CPU `AFTimeBlockEncoder::Decode`.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void dequantizeKernel(const unsigned int *symbols, size_t nVis,
                                 size_t visPerRow, size_t nPol, size_t nAnt,
                                 const float *meta, const int *ant1,
                                 const int *ant2, const float *decValues,
                                 size_t decCount, float *out) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nVis) return;
  const size_t row = i / visPerRow;
  const size_t rem = i % visPerRow;
  const size_t ch = rem / nPol;
  const size_t pol = rem % nPol;
  const double chRms = static_cast<double>(meta[ch * nPol + pol]);
  const double antFac =
      static_cast<double>(meta[visPerRow + pol * nAnt + ant1[row]]) *
      static_cast<double>(meta[visPerRow + pol * nAnt + ant2[row]]);
  const double factor = chRms * antFac;
  const unsigned int sRe = symbols[i * 2];
  const unsigned int sIm = symbols[i * 2 + 1];
  const double vRe =
      (sRe < decCount) ? static_cast<double>(decValues[sRe]) : nan("");
  const double vIm =
      (sIm < decCount) ? static_cast<double>(decValues[sIm]) : nan("");
  out[i * 2] = static_cast<float>(vRe * factor);
  out[i * 2 + 1] = static_cast<float>(vIm * factor);
}

}  // namespace

void d2_dequantize(const unsigned int *symbols, size_t nRows, size_t nChannels,
                   size_t nPol, size_t nAnt, const float *meta,
                   const int *ant1, const int *ant2, const float *decValues,
                   size_t decCount, float *interleaved,
                   dysco_stream_t stream) {
  const size_t visPerRow = nChannels * nPol;
  const size_t nVis = nRows * visPerRow;
  const unsigned threads = 256;
  const unsigned blocks =
      static_cast<unsigned>((nVis + threads - 1) / threads);
  DYSCO_LAUNCH_1D(dequantizeKernel, blocks, threads, stream, symbols, nVis,
                  visPerRow, nPol, nAnt, meta, ant1, ant2, decValues, decCount,
                  interleaved);
}

}  // namespace gpu
}  // namespace dyscostman
