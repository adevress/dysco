#ifndef DYSCO_CUDA_KERNELS_H
#define DYSCO_CUDA_KERNELS_H

/**
 * @file
 * Host-side launchers for the Dysco GPU kernels.  Every kernel lives in its
 * own `.cu` file; this header is the single place where the orchestration
 * (`gpu_af_codec.cu`) sees them.
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

#include <cstddef>
#include <cstdint>

namespace dyscostman {
namespace gpu {

// ---------------------------------------------------------------- k1_gather_convert
void k1_gather_convert(const float *interleaved, size_t nRows, size_t visPerRow,
                       double *re, double *im, dysco_stream_t stream);

// ---------------------------------------------------------------- k2_channel_rms
void k2_channel_rms(const double *re, const double *im, size_t nRows,
                    size_t visPerRow, float *channelRms,
                    dysco_stream_t stream);

// ---------------------------------------------------------------- k3_antenna_rms_matrix
void k3_antenna_rms_matrix(const double *re, const double *im, const int *ant1,
                           const int *ant2, size_t nRows, size_t nChannels,
                           size_t nPol, size_t nAnt, size_t pol,
                           double *matrix, dysco_stream_t stream);

// ---------------------------------------------------------------- k4_antenna_gain_solve
void k4_antenna_gain_solve(const double *matrix, size_t nAnt,
                           double *rmsPerAntenna, dysco_stream_t stream);
/// Convert the double-precision gains into the float metadata region.
void k4_store_antenna_gains(const double *rmsPerAntenna, size_t nAnt,
                            float *destination, dysco_stream_t stream);

// ---------------------------------------------------------------- k5_apply_norm
void k5a_apply_channel_norm(double *re, double *im, size_t nVis,
                            size_t visPerRow, const float *channelRms,
                            dysco_stream_t stream);
void k5b_apply_antenna_norm(double *re, double *im, const int *ant1,
                            const int *ant2, size_t nRows, size_t nChannels,
                            size_t nPol, size_t pol,
                            const double *rmsPerAntenna, dysco_stream_t stream);

// ---------------------------------------------------------------- k6_fit_to_max
/// Runs the complete greedy fit-to-maximum optimization for all
/// polarizations.  `maxLevel` is the largest decodable value.
void k6_fit_to_max(double *re, double *im, const int *ant1, const int *ant2,
                   size_t nRows, size_t nChannels, size_t nPol, size_t nAnt,
                   float *meta, double maxLevel, dysco_stream_t stream);

// ---------------------------------------------------------------- k7_quantize_dither
void k7_quantize_dither(const double *re, const double *im, size_t nVis,
                        const float *decValues, size_t decCount, uint32_t seedLo,
                        uint32_t seedHi, uint64_t blockIndex,
                        unsigned int *symbols, dysco_stream_t stream);

// ---------------------------------------------------------------- k8_bit_pack
void k8_bit_pack(const unsigned int *symbols, size_t symbolCount, unsigned bits,
                 unsigned char *packed, size_t packedBytes,
                 dysco_stream_t stream);

// ---------------------------------------------------------------- d1_bit_unpack
void d1_bit_unpack(const unsigned char *packed, size_t packedBytes,
                   unsigned bits, unsigned int *symbols, size_t symbolCount,
                   dysco_stream_t stream);

// ---------------------------------------------------------------- d2_dequantize
void d2_dequantize(const unsigned int *symbols, size_t nRows, size_t nChannels,
                   size_t nPol, size_t nAnt, const float *meta,
                   const int *ant1, const int *ant2, const float *decValues,
                   size_t decCount, float *interleaved, dysco_stream_t stream);

}  // namespace gpu
}  // namespace dyscostman

#endif  // DYSCO_CUDA_KERNELS_H
