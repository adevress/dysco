// K1 — gather + layout transform + float->double widening.
//
// The measurement set (and the in-memory TimeBlockBuffer) stores each row as a
// contiguous array of interleaved complex<float> values (re, im, re, im, ...),
// while all subsequent kernels want a structure-of-arrays layout in double
// precision.  This kernel does the AoS -> SoA transform.  The antenna index
// arrays are already resident in managed memory and are not touched here.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void gatherConvertKernel(const float *interleaved, std::size_t nVis,
                                    double *re, double *im) {
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       i < nVis;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    // The input pointer is float-based: complex<float> is 2 floats.
    re[i] = static_cast<double>(interleaved[i * 2]);
    im[i] = static_cast<double>(interleaved[i * 2 + 1]);
  }
}

}  // namespace

void k1_gather_convert(dysco::span<const float> interleaved, std::size_t nRows,
                       std::size_t visPerRow, dysco::span<double> re,
                       dysco::span<double> im, dysco_stream_t stream) {
  const std::size_t nVis = nRows * visPerRow;
  constexpr unsigned kThreads = 256;
  const auto blocks =
      static_cast<unsigned>((nVis + kThreads - 1) / kThreads);
  DYSCO_LAUNCH_1D(gatherConvertKernel, blocks, kThreads, stream,
                  interleaved.data(), nVis, re.data(), im.data());
}

}  // namespace gpu
}  // namespace dyscostman
