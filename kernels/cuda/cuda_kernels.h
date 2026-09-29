#ifndef DYSCO_CUDA_KERNELS_H
#define DYSCO_CUDA_KERNELS_H

/**
 * @file
 * Host-side launchers for the Dysco GPU kernels.  Every kernel lives in its
 * own `.cu` file; this header is the single place where the orchestration
 * (`gpu_af_codec.cu`) sees them.
 *
 * The host API (and the device kernels behind it) is expressed in terms of
 * libcu++ views — `cuda::std::span` for one-dimensional buffers and
 * `cuda::std::mdspan` (via `MdSpan2D`) for row-major two-dimensional arrays —
 * instead of raw pointer/size pairs.  The views are `__host__ __device__`
 * types and are passed to the kernels by value.
 *
 * Data layout used by all kernels (SoA form of one AF time block):
 *
 *   visible count            nVis = nRows * visPerRow,  visPerRow = nChannels*nPol
 *   interleaved              MdSpan2D<const float>[nVis, 2]
 *   re/im                    MdSpan2D<double>[nRows, visPerRow]
 *                            index = (row, ch*nPol + pol)
 *   ant1/ant2                span<int>[nRows]
 *   channelRms               span<float>[visPerRow]
 *   matrix                   MdSpan2D<double>[nAnt, nAnt]
 *   antennaRms               span<float>[nPol*nAnt]  (meta order: [pol*nAnt + ant])
 *   symbols                  MdSpan2D<unsigned int>[nVis, 2]  (re, imag)
 *   packed                   span<unsigned char>[ceil(2*nVis*bits/8)]
 */

#include "../gpu_portability.h"
#include "cuda_cccl.cuh"

#include <cstddef>
#include <cstdint>

namespace dyscostman {
namespace gpu {

/// Default 1-D block size for the simple element-wise kernels.
inline constexpr unsigned kThreadBlockSize = 256;

/// Number of blocks needed to cover `n` elements with `threads` threads, at
/// least one.  All kernels are grid-stride or bounds-checked, so an
/// over-estimate is harmless.
[[nodiscard]] inline unsigned gridFor(std::size_t n,
                                      unsigned threads) noexcept {
  const std::size_t blocks = (n + threads - 1) / threads;
  return static_cast<unsigned>(blocks == 0 ? 1 : blocks);
}

// ---------------------------------------------------------------- k1_gather_convert
void k1_gather_convert(MdSpan2D<const float> interleaved,
                       cuda::std::span<double> re, cuda::std::span<double> im,
                       dysco_stream_t stream);

// ---------------------------------------------------------------- k2_channel_rms
void k2_channel_rms(MdSpan2D<const double> re, MdSpan2D<const double> im,
                    cuda::std::span<float> channelRms, dysco_stream_t stream);

// ---------------------------------------------------------------- k3_antenna_rms_matrix
void k3_antenna_rms_matrix(MdSpan2D<const double> re,
                           MdSpan2D<const double> im,
                           cuda::std::span<const int> ant1,
                           cuda::std::span<const int> ant2,
                           std::size_t nChannels, std::size_t nPol,
                           std::size_t pol, MdSpan2D<double> matrix,
                           dysco_stream_t stream);

// ---------------------------------------------------------------- k4_antenna_gain_solve
void k4_antenna_gain_solve(MdSpan2D<const double> matrix,
                           cuda::std::span<double> rmsPerAntenna,
                           dysco_stream_t stream);
/// Convert the double-precision gains into the float metadata region.
void k4_store_antenna_gains(cuda::std::span<const double> rmsPerAntenna,
                            cuda::std::span<float> destination,
                            dysco_stream_t stream);

// ---------------------------------------------------------------- k5_apply_norm
void k5a_apply_channel_norm(cuda::std::span<double> re,
                            cuda::std::span<double> im, std::size_t visPerRow,
                            cuda::std::span<const float> channelRms,
                            dysco_stream_t stream);
void k5b_apply_antenna_norm(MdSpan2D<double> re, MdSpan2D<double> im,
                            cuda::std::span<const int> ant1,
                            cuda::std::span<const int> ant2,
                            std::size_t nChannels, std::size_t nPol,
                            std::size_t pol,
                            cuda::std::span<const double> rmsPerAntenna,
                            dysco_stream_t stream);

// ---------------------------------------------------------------- k6_fit_to_max
/// Runs the fit-to-maximum optimization for all polarizations.  `maxLevel` is
/// the largest decodable value.  The default algorithm is the parallel
/// damped-simultaneous ("Jacobi") variant; the legacy greedy variant
/// (bit-identical to the CPU encoder) is selected with the `DYSCO_K6_GREEDY`
/// environment variable.
void k6_fit_to_max(MdSpan2D<double> re, MdSpan2D<double> im,
                   cuda::std::span<const int> ant1,
                   cuda::std::span<const int> ant2, std::size_t nChannels,
                   std::size_t nPol, std::size_t nAnt,
                   cuda::std::span<float> meta, double maxLevel,
                   dysco_stream_t stream);

// ---------------------------------------------------------------- k7_quantize_dither
/// Flat quantization used by the unit tests (one symbol per value).
void k7_quantize_flat(cuda::std::span<const float> values,
                      cuda::std::span<const float> decValues,
                      std::uint32_t seedLo, std::uint32_t seedHi,
                      std::uint64_t blockIndex,
                      cuda::std::span<unsigned int> symbols,
                      dysco_stream_t stream);
void k7_quantize_dither(MdSpan2D<const double> re, MdSpan2D<const double> im,
                        cuda::std::span<const float> decValues,
                        std::uint32_t seedLo, std::uint32_t seedHi,
                        std::uint64_t blockIndex,
                        cuda::std::span<unsigned int> symbols,
                        dysco_stream_t stream);

// ---------------------------------------------------------------- k8_bit_pack
void k8_bit_pack(cuda::std::span<const unsigned int> symbols, unsigned bits,
                 cuda::std::span<unsigned char> packed, dysco_stream_t stream);

// ---------------------------------------------------------------- d1_bit_unpack
void d1_bit_unpack(cuda::std::span<const unsigned char> packed, unsigned bits,
                   cuda::std::span<unsigned int> symbols,
                   dysco_stream_t stream);

// ---------------------------------------------------------------- d2_dequantize
void d2_dequantize(MdSpan2D<const unsigned int> symbols,
                   cuda::std::span<const float> meta,
                   cuda::std::span<const int> ant1,
                   cuda::std::span<const int> ant2, std::size_t nRows,
                   std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                   cuda::std::span<const float> decValues,
                   MdSpan2D<float> interleaved, dysco_stream_t stream);

}  // namespace gpu
}  // namespace dyscostman

#endif  // DYSCO_CUDA_KERNELS_H
