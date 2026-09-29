// D2 — dequantization / decoding.
//
// value = decValues[symbol] * channelRms[ch,pol] * g_a1 * g_a2
// with the NaN symbol (index == decCount) producing a NaN visibility, exactly
// like the CPU `AFTimeBlockEncoder::Decode`.
//
// The kernel is launched with a 2D grid: `blockIdx.x` selects the row and
// `blockIdx.y` the (channel,pol) within the row, which avoids a per-element
// 64-bit `i / visPerRow` division; the remaining index arithmetic is done in
// 32 bits.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void dequantizeKernel(MdSpan2D<const unsigned int> symbols,
                                 cuda::std::span<const float> meta,
                                 cuda::std::span<const int> ant1,
                                 cuda::std::span<const int> ant2,
                                 std::size_t visPerRow, unsigned nPol,
                                 std::size_t nAnt,
                                 cuda::std::span<const float> decValues,
                                 MdSpan2D<float> interleaved) {
  const unsigned decCount = static_cast<unsigned>(decValues.size());
  const double nanValue = maxKeyToDouble(kQuietNaNBits);
  const std::size_t row = blockIdx.x;
  const unsigned rem =
      static_cast<unsigned>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (rem >= visPerRow) return;
  const unsigned ch = rem / nPol;
  const unsigned pol = rem % nPol;
  const std::size_t i = row * visPerRow + rem;
  const double chRms = static_cast<double>(meta[ch * nPol + pol]);
  const double antFac =
      static_cast<double>(meta[visPerRow + pol * nAnt + ant1[row]]) *
      static_cast<double>(meta[visPerRow + pol * nAnt + ant2[row]]);
  const double factor = chRms * antFac;
  const unsigned int sRe = symbols(i, 0);
  const unsigned int sIm = symbols(i, 1);
  const double vRe =
      (sRe < decCount) ? static_cast<double>(decValues[sRe]) : nanValue;
  const double vIm =
      (sIm < decCount) ? static_cast<double>(decValues[sIm]) : nanValue;
  interleaved(i, 0) = static_cast<float>(vRe * factor);
  interleaved(i, 1) = static_cast<float>(vIm * factor);
}

}  // namespace

void d2_dequantize(MdSpan2D<const unsigned int> symbols,
                   cuda::std::span<const float> meta,
                   cuda::std::span<const int> ant1,
                   cuda::std::span<const int> ant2, std::size_t nRows,
                   std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                   cuda::std::span<const float> decValues,
                   MdSpan2D<float> interleaved, dysco_stream_t stream) {
  const std::size_t visPerRow = nChannels * nPol;
  constexpr unsigned kThreads = kThreadBlockSize;
  const dim3 grid(static_cast<unsigned>(nRows),
                  gridFor(visPerRow, kThreads));
  DYSCO_LAUNCH_1D(dequantizeKernel, grid, kThreads, stream, symbols, meta, ant1,
                  ant2, visPerRow, static_cast<unsigned>(nPol), nAnt,
                  decValues, interleaved);
}

}  // namespace gpu
}  // namespace dyscostman
