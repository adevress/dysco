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

__global__ void gatherConvertKernel(MdSpan2D<const float> interleaved,
                                    cuda::std::span<double> re,
                                    cuda::std::span<double> im) {
  const std::size_t nVis = interleaved.extent(0);
  const std::size_t stride = gridStrideSize();
  for (std::size_t i = threadFlattenedId(); i < nVis; i += stride) {
    // A complex<float> is two floats; the mdspan's second extent is 2.
    re[i] = static_cast<double>(interleaved(i, 0));
    im[i] = static_cast<double>(interleaved(i, 1));
  }
}

}  // namespace

void k1_gather_convert(MdSpan2D<const float> interleaved,
                       cuda::std::span<double> re, cuda::std::span<double> im,
                       dysco_stream_t stream) {
  const std::size_t nVis = interleaved.extent(0);
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  const auto blocks = gridFor(nVis, kThreads);
  DYSCO_LAUNCH_1D(gatherConvertKernel, blocks, kThreads, stream, interleaved,
                  re, im);
}

}  // namespace gpu
}  // namespace dyscostman
