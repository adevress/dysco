// Orchestration of the AF GPU codec: workspace management (CUDA managed
// memory / transparent page migration) and the fixed pipeline of kernel
// launches.  No computation happens here; this file only wires the individual
// per-kernel files together.

#include "gpu_af_codec.h"

#include "cuda_kernels.h"
#include "../gpu_portability.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace dyscostman {
namespace gpu {

bool available() {
  int n = 0;
  cudaError_t err = cudaGetDeviceCount(&n);
  if (err != cudaSuccess)
    std::fprintf(stderr, "[dysco-gpu] cudaGetDeviceCount: %s\n",
                 cudaGetErrorString(err));
  return err == cudaSuccess && n > 0;
}

const char *deviceName() { return dysco_gpu::device_name(); }

namespace {
size_t packedByteCountFor(size_t symbolCount, unsigned bits) {
  return (symbolCount * bits + 7) / 8;
}
}  // namespace

struct AfCodec::Impl {
  size_t nRows, nChannels, nPol, nAnt, visPerRow, nVis, decCount, bits;
  size_t packedBytes;
  float *decValues = nullptr;
  float *inInterleaved = nullptr;  // 2*nVis
  double *re = nullptr, *im = nullptr;
  int *ant1 = nullptr, *ant2 = nullptr;
  float *meta = nullptr;  // visPerRow + nPol*nAnt
  double *matrix = nullptr;
  double *rmsPerAnt = nullptr;
  unsigned int *symbols = nullptr;
  unsigned char *packed = nullptr;
  float *out = nullptr;

  Impl(size_t rows, size_t channels, size_t pol, size_t ant,
       const float *decVals, size_t dec, unsigned bitsPerSymbol)
      : nRows(rows),
        nChannels(channels),
        nPol(pol),
        nAnt(ant),
        visPerRow(channels * pol),
        nVis(rows * channels * pol),
        decCount(dec),
        bits(bitsPerSymbol),
        packedBytes(packedByteCountFor(2 * nVis, bitsPerSymbol)) {
    DYSCO_MANAGED_ALLOC(&decValues, decCount * sizeof(float));
    DYSCO_MANAGED_ALLOC(&inInterleaved, 2 * nVis * sizeof(float));
    DYSCO_MANAGED_ALLOC(&re, nVis * sizeof(double));
    DYSCO_MANAGED_ALLOC(&im, nVis * sizeof(double));
    DYSCO_MANAGED_ALLOC(&ant1, nRows * sizeof(int));
    DYSCO_MANAGED_ALLOC(&ant2, nRows * sizeof(int));
    DYSCO_MANAGED_ALLOC(&meta, (visPerRow + nPol * nAnt) * sizeof(float));
    DYSCO_MANAGED_ALLOC(&matrix, nAnt * nAnt * sizeof(double));
    DYSCO_MANAGED_ALLOC(&rmsPerAnt, nAnt * sizeof(double));
    DYSCO_MANAGED_ALLOC(&symbols, 2 * nVis * sizeof(unsigned int));
    DYSCO_MANAGED_ALLOC(&packed, packedBytes * sizeof(unsigned char));
    DYSCO_MANAGED_ALLOC(&out, 2 * nVis * sizeof(float));
    std::memcpy(decValues, decVals, decCount * sizeof(float));
  }

  ~Impl() {
    DYSCO_FREE(decValues);
    DYSCO_FREE(inInterleaved);
    DYSCO_FREE(re);
    DYSCO_FREE(im);
    DYSCO_FREE(ant1);
    DYSCO_FREE(ant2);
    DYSCO_FREE(meta);
    DYSCO_FREE(matrix);
    DYSCO_FREE(rmsPerAnt);
    DYSCO_FREE(symbols);
    DYSCO_FREE(packed);
    DYSCO_FREE(out);
  }

  double maxLevel() const {
    return decCount == 0 ? 0.0 : static_cast<double>(decValues[decCount - 1]);
  }
};

AfCodec::AfCodec(size_t nRows, size_t nChannels, size_t nPol, size_t nAnt,
                 const float *decValues, size_t decCount,
                 unsigned bitsPerSymbol)
    : _impl(new Impl(nRows, nChannels, nPol, nAnt, decValues, decCount,
                     bitsPerSymbol)) {}

AfCodec::~AfCodec() = default;

size_t AfCodec::packedByteCount() const { return _impl->packedBytes; }

void AfCodec::encode(const float *interleavedIn, const int *ant1,
                     const int *ant2, uint64_t seed, uint64_t blockIndex,
                     bool fitToMaximum, float *metaOut,
                     unsigned char *packedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;

  // Upload the block (transparent page migration: a plain CPU write to the
  // managed buffer is enough; pages move on first device access).
  std::memcpy(s.inInterleaved, interleavedIn, 2 * s.nVis * sizeof(float));
  std::memcpy(s.ant1, ant1, s.nRows * sizeof(int));
  std::memcpy(s.ant2, ant2, s.nRows * sizeof(int));

  k1_gather_convert(s.inInterleaved, s.nRows, s.visPerRow, s.re, s.im, stream);

  k2_channel_rms(s.re, s.im, s.nRows, s.visPerRow, s.meta, stream);
  k5a_apply_channel_norm(s.re, s.im, s.nVis, s.visPerRow, s.meta, stream);

  for (size_t pol = 0; pol != s.nPol; ++pol) {
    k3_antenna_rms_matrix(s.re, s.im, s.ant1, s.ant2, s.nRows, s.nChannels,
                          s.nPol, s.nAnt, pol, s.matrix, stream);
    k4_antenna_gain_solve(s.matrix, s.nAnt, s.rmsPerAnt, stream);
    k5b_apply_antenna_norm(s.re, s.im, s.ant1, s.ant2, s.nRows, s.nChannels,
                           s.nPol, pol, s.rmsPerAnt, stream);
    // Store the antenna gains in the metadata block, in the same order as the
    // CPU encoder ([visPerRow + pol*nAnt + antenna]).
    k4_store_antenna_gains(s.rmsPerAnt, s.nAnt,
                           s.meta + s.visPerRow + pol * s.nAnt, stream);
  }

  if (fitToMaximum) {
    k6_fit_to_max(s.re, s.im, s.ant1, s.ant2, s.nRows, s.nChannels, s.nPol,
                  s.nAnt, s.meta, s.maxLevel(), stream);
  }

  const uint32_t seedLo = static_cast<uint32_t>(seed);
  const uint32_t seedHi = static_cast<uint32_t>(seed >> 32);
  k7_quantize_dither(s.re, s.im, s.nVis, s.decValues, s.decCount, seedLo,
                     seedHi, blockIndex, s.symbols, stream);
  k8_bit_pack(s.symbols, 2 * s.nVis, static_cast<unsigned>(s.bits), s.packed,
              s.packedBytes, stream);

  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(metaOut, s.meta, (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(packedOut, s.packed, s.packedBytes);
}

void AfCodec::decode(const unsigned char *packedIn, size_t packedBytes,
                     const float *metaIn, const int *ant1, const int *ant2,
                     float *interleavedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;
  if (packedBytes > s.packedBytes)
    throw std::runtime_error("Dysco GPU decode: packed buffer too small");

  std::memcpy(s.packed, packedIn, packedBytes);
  std::memcpy(s.meta, metaIn, (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(s.ant1, ant1, s.nRows * sizeof(int));
  std::memcpy(s.ant2, ant2, s.nRows * sizeof(int));

  d1_bit_unpack(s.packed, packedBytes, static_cast<unsigned>(s.bits), s.symbols,
                2 * s.nVis, stream);
  d2_dequantize(s.symbols, s.nRows, s.nChannels, s.nPol, s.nAnt, s.meta, s.ant1,
                s.ant2, s.decValues, s.decCount, s.out, stream);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(interleavedOut, s.out, 2 * s.nVis * sizeof(float));
}

}  // namespace gpu
}  // namespace dyscostman
