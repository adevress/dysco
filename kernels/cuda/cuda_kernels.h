#ifndef DYSCO_CUDA_KERNELS_H
#define DYSCO_CUDA_KERNELS_H

/**
 * @file
 * Host-side launchers for the Dysco GPU kernels.  Every kernel lives in its
 * own `.cu` file; this header is the single place where the orchestration
 * (`gpu_af_codec.cu`) sees them.
 *
 * The host API is expressed in terms of `dysco::span` views (which are
 * `std::span` under C++20) instead of raw pointer/size pairs.  The device
 * kernels themselves still take plain pointers, since spans are not
 * device-annotated; the launchers forward `data()` and `size()`.
 *
 * Data layout used by all kernels (SoA form of one AF time block):
 *
 *   visible count            nVis = nRows * visPerRow,  visPerRow = nChannels*nPol
 *   re/im                    double[nVis], index = row*visPerRow + ch*nPol + pol
 *   ant1/ant2                int[nRows]
 *   channelRms               float[visPerRow]
 *   antennaRms               float[nPol*nAnt]  (meta order: [pol*nAnt + ant])
 *   symbols                  uint32_t[2*nVis]  (re at 2i, imag at 2i+1)
 *   packed                   uint8_t[ceil(2*nVis*bits/8)]
 */

#include "../gpu_portability.h"
#include "../span.h"

#include <cstddef>
#include <cstdint>

namespace dyscostman::gpu {

// ---------------------------------------------------------------- k1_gather_convert
void k1_gather_convert(dysco::span<const float> interleaved, std::size_t nRows,
                       std::size_t visPerRow, dysco::span<double> re,
                       dysco::span<double> im, dysco_stream_t stream);

// ---------------------------------------------------------------- k2_channel_rms
void k2_channel_rms(dysco::span<const double> re, dysco::span<const double> im,
                    std::size_t nRows, std::size_t visPerRow,
                    dysco::span<float> channelRms, dysco_stream_t stream);

// ---------------------------------------------------------------- k3_antenna_rms_matrix
void k3_antenna_rms_matrix(dysco::span<const double> re,
                           dysco::span<const double> im,
                           dysco::span<const int> ant1,
                           dysco::span<const int> ant2, std::size_t nRows,
                           std::size_t nChannels, std::size_t nPol,
                           std::size_t nAnt, std::size_t pol,
                           dysco::span<double> matrix, dysco_stream_t stream);

// ---------------------------------------------------------------- k4_antenna_gain_solve
void k4_antenna_gain_solve(dysco::span<const double> matrix, std::size_t nAnt,
                           dysco::span<double> rmsPerAntenna,
                           dysco_stream_t stream);
/// Convert the double-precision gains into the float metadata region.
void k4_store_antenna_gains(dysco::span<const double> rmsPerAntenna,
                            dysco::span<float> destination,
                            dysco_stream_t stream);

// ---------------------------------------------------------------- k5_apply_norm
void k5a_apply_channel_norm(dysco::span<double> re, dysco::span<double> im,
                            std::size_t visPerRow,
                            dysco::span<const float> channelRms,
                            dysco_stream_t stream);
void k5b_apply_antenna_norm(dysco::span<double> re, dysco::span<double> im,
                            dysco::span<const int> ant1,
                            dysco::span<const int> ant2, std::size_t nRows,
                            std::size_t nChannels, std::size_t nPol,
                            std::size_t pol,
                            dysco::span<const double> rmsPerAntenna,
                            dysco_stream_t stream);

// ---------------------------------------------------------------- k6_fit_to_max
/// Runs the complete greedy fit-to-maximum optimization for all
/// polarizations.  `maxLevel` is the largest decodable value.
void k6_fit_to_max(dysco::span<double> re, dysco::span<double> im,
                   dysco::span<const int> ant1, dysco::span<const int> ant2,
                   std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                   std::size_t nAnt, dysco::span<float> meta, double maxLevel,
                   dysco_stream_t stream);

// ---------------------------------------------------------------- k7_quantize_dither
/// Flat quantization used by the unit tests (one symbol per value).
void k7_quantize_flat(dysco::span<const float> values,
                      dysco::span<const float> decValues, std::uint32_t seedLo,
                      std::uint32_t seedHi, std::uint64_t blockIndex,
                      dysco::span<unsigned int> symbols,
                      dysco_stream_t stream);
void k7_quantize_dither(dysco::span<const double> re,
                        dysco::span<const double> im,
                        dysco::span<const float> decValues, std::uint32_t seedLo,
                        std::uint32_t seedHi, std::uint64_t blockIndex,
                        dysco::span<unsigned int> symbols,
                        dysco_stream_t stream);

// ---------------------------------------------------------------- k8_bit_pack
void k8_bit_pack(dysco::span<const unsigned int> symbols, unsigned bits,
                 dysco::span<unsigned char> packed, dysco_stream_t stream);

// ---------------------------------------------------------------- d1_bit_unpack
void d1_bit_unpack(dysco::span<const unsigned char> packed, unsigned bits,
                   dysco::span<unsigned int> symbols, dysco_stream_t stream);

// ---------------------------------------------------------------- d2_dequantize
void d2_dequantize(dysco::span<const unsigned int> symbols, std::size_t nRows,
                   std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                   dysco::span<const float> meta, dysco::span<const int> ant1,
                   dysco::span<const int> ant2,
                   dysco::span<const float> decValues,
                   dysco::span<float> interleaved, dysco_stream_t stream);

}  // namespace dyscostman::gpu

#endif  // DYSCO_CUDA_KERNELS_H
