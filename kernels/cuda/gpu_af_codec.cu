// Orchestration of the AF GPU codec: workspace management (CUDA managed
// memory / transparent page migration) and the fixed pipeline of kernel
// launches.  No computation happens here; this file only wires the individual
// per-kernel files together.
//
// The managed buffers are exposed to the kernels as libcu++ views
// (`cuda::std::span` / `cuda::std::mdspan`); this file is the only place where
// the raw allocation pointers are turned into those views.

#include "gpu_af_codec.h"

#include "../gpu_portability.h"
#include "cuda_kernels.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace dyscostman::gpu {

bool available() {
  std::int32_t count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess)
    std::fprintf(stderr, "[dysco-gpu] cudaGetDeviceCount: %s\n",
                 cudaGetErrorString(error));
  return error == cudaSuccess && count > 0;
}

std::string_view deviceName() { return dysco_gpu::device_name(); }

namespace {
[[nodiscard]] constexpr std::size_t packedByteCountFor(
    std::size_t symbolCount, std::uint32_t bits) noexcept {
  return (symbolCount * bits + 7) / 8;
}

template <class T>
using ManagedBuffer = std::unique_ptr<T, dysco_gpu::ManagedDeleter>;

template <class T>
[[nodiscard]] ManagedBuffer<T> allocateManaged(std::size_t count) {
  T *raw = nullptr;
  DYSCO_MANAGED_ALLOC(&raw, count * sizeof(T));
  return ManagedBuffer<T>(raw);
}
}  // namespace

struct AfCodec::Impl {
  template <class T>
  using Managed = std::unique_ptr<T, dysco_gpu::ManagedDeleter>;

  template <class T>
  [[nodiscard]] static Managed<T> allocate(std::size_t count) {
    T *raw = nullptr;
    DYSCO_MANAGED_ALLOC(&raw, count * sizeof(T));
    return Managed<T>(raw);
  }

  const std::size_t nRows;
  const std::size_t nChannels;
  const std::size_t nPol;
  const std::size_t nAnt;
  const std::size_t visPerRow;
  const std::size_t nVis;
  const std::size_t decCount;
  const std::uint32_t bits;
  const std::size_t packedBytes;

  Managed<float> decValues;
  Managed<float> inInterleaved;  // 2*nVis
  Managed<double> re, im;
  Managed<std::int32_t> ant1, ant2;
  Managed<float> meta;  // visPerRow + nPol*nAnt
  Managed<double> matrix;
  Managed<double> rmsPerAnt;
  Managed<double> rmsScratch;  // K4 'next' vector (nAnt > block threads)
  Managed<std::uint32_t> symbols;
  Managed<std::uint8_t> packed;
  Managed<float> out;

  Impl(std::size_t rows, std::size_t channels, std::size_t pol,
       std::size_t antennas, dysco::span<const float> decValuesIn,
       std::uint32_t bitsPerSymbol)
      : nRows(rows),
        nChannels(channels),
        nPol(pol),
        nAnt(antennas),
        visPerRow(channels * pol),
        nVis(rows * channels * pol),
        decCount(decValuesIn.size()),
        bits(bitsPerSymbol),
        packedBytes(packedByteCountFor(2 * nVis, bitsPerSymbol)),
        decValues(allocate<float>(decCount)),
        inInterleaved(allocate<float>(2 * nVis)),
        re(allocate<double>(nVis)),
        im(allocate<double>(nVis)),
        ant1(allocate<std::int32_t>(nRows)),
        ant2(allocate<std::int32_t>(nRows)),
        meta(allocate<float>(visPerRow + nPol * nAnt)),
        matrix(allocate<double>(nAnt * nAnt)),
        rmsPerAnt(allocate<double>(nAnt)),
        rmsScratch(allocate<double>(nAnt)),
        symbols(allocate<std::uint32_t>(2 * nVis)),
        packed(allocate<std::uint8_t>(packedBytes)),
        out(allocate<float>(2 * nVis)) {
    std::copy(decValuesIn.begin(), decValuesIn.end(), decValues.get());
  }

  [[nodiscard]] double maxLevel() const noexcept {
    return decCount == 0 ? 0.0 : static_cast<double>(decValues.get()[decCount - 1]);
  }

  // Row-major 2-D views of the managed workspaces, as the kernels expect them.
  [[nodiscard]] MdSpan2D<double> reMatrix() const noexcept {
    return MdSpan2D<double>(re.get(), nRows, visPerRow);
  }
  [[nodiscard]] MdSpan2D<double> imMatrix() const noexcept {
    return MdSpan2D<double>(im.get(), nRows, visPerRow);
  }
  /// Interleaved (re, im) per visibility, shape [nVis, 2].
  [[nodiscard]] MdSpan2D<std::uint32_t> symbolsMatrix() const noexcept {
    return MdSpan2D<std::uint32_t>(symbols.get(), nVis, 2);
  }
  [[nodiscard]] MdSpan2D<float> outMatrix() const noexcept {
    return MdSpan2D<float>(out.get(), nVis, 2);
  }
  [[nodiscard]] MdSpan2D<const float> inInterleavedMatrix() const noexcept {
    return MdSpan2D<const float>(inInterleaved.get(), nVis, 2);
  }
};

AfCodec::AfCodec(std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                 std::size_t nAnt, dysco::span<const float> decValues,
                 std::uint32_t bitsPerSymbol)
    : _impl(std::make_unique<Impl>(nRows, nChannels, nPol, nAnt, decValues,
                                   bitsPerSymbol)) {}

AfCodec::~AfCodec() = default;

std::size_t AfCodec::packedByteCount() const noexcept {
  return _impl->packedBytes;
}

void AfCodec::encode(dysco::span<const float> interleavedIn,
                     dysco::span<const std::int32_t> ant1,
                     dysco::span<const std::int32_t> ant2,
                     std::uint64_t seed, std::uint64_t blockIndex,
                     bool fitToMaximum, dysco::span<float> metaOut,
                     dysco::span<std::uint8_t> packedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;
  const cuda::std::span<float> metaView(s.meta.get(),
                                        s.visPerRow + s.nPol * s.nAnt);
  const cuda::std::span<const std::int32_t> ant1View(s.ant1.get(), s.nRows);
  const cuda::std::span<const std::int32_t> ant2View(s.ant2.get(), s.nRows);

  // Upload the block (transparent page migration: a plain CPU write to the
  // managed buffer is enough; pages move on first device access).
  std::memcpy(s.inInterleaved.get(), interleavedIn.data(),
              2 * s.nVis * sizeof(float));
  std::memcpy(s.ant1.get(), ant1.data(), s.nRows * sizeof(std::int32_t));
  std::memcpy(s.ant2.get(), ant2.data(), s.nRows * sizeof(std::int32_t));

  k1_gather_convert(s.inInterleavedMatrix(),
                    cuda::std::span<double>(s.re.get(), s.nVis),
                    cuda::std::span<double>(s.im.get(), s.nVis), stream);

  const MdSpan2D<double> re = s.reMatrix();
  const MdSpan2D<double> im = s.imMatrix();

  k2_channel_rms(re, im,
                 cuda::std::span<float>(s.meta.get(), s.visPerRow), stream);
  k5a_apply_channel_norm(cuda::std::span<double>(s.re.get(), s.nVis),
                         cuda::std::span<double>(s.im.get(), s.nVis),
                         s.visPerRow,
                         cuda::std::span<const float>(s.meta.get(),
                                                      s.visPerRow),
                         stream);

  for (std::size_t pol = 0; pol != s.nPol; ++pol) {
    k3_antenna_rms_matrix(re, im, ant1View, ant2View, s.nChannels, s.nPol,
                          pol, MdSpan2D<double>(s.matrix.get(), s.nAnt, s.nAnt),
                          stream);
    k4_antenna_gain_solve(MdSpan2D<const double>(s.matrix.get(), s.nAnt,
                                                 s.nAnt),
                          cuda::std::span<double>(s.rmsPerAnt.get(), s.nAnt),
                          cuda::std::span<double>(s.rmsScratch.get(), s.nAnt),
                          stream);
    k5b_apply_antenna_norm(re, im, ant1View, ant2View, s.nChannels, s.nPol,
                           pol,
                           cuda::std::span<const double>(s.rmsPerAnt.get(),
                                                         s.nAnt),
                           stream);
    // Store the antenna gains in the metadata block, in the same order as the
    // CPU encoder ([visPerRow + pol*nAnt + antenna]).
    k4_store_antenna_gains(
        cuda::std::span<const double>(s.rmsPerAnt.get(), s.nAnt),
        cuda::std::span<float>(s.meta.get() + s.visPerRow + pol * s.nAnt,
                               s.nAnt),
        stream);
  }

  if (fitToMaximum) {
    k6_fit_to_max(re, im, ant1View, ant2View, s.nChannels, s.nPol, s.nAnt,
                  metaView, s.maxLevel(), stream);
  }

  k7_quantize_dither(re, im,
                     cuda::std::span<const float>(s.decValues.get(), s.decCount),
      static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
      blockIndex, cuda::std::span<std::uint32_t>(s.symbols.get(), 2 * s.nVis),
      stream);
  k8_bit_pack(
      cuda::std::span<const std::uint32_t>(s.symbols.get(), 2 * s.nVis),
      s.bits, cuda::std::span<std::uint8_t>(s.packed.get(), s.packedBytes),
      stream);

  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(metaOut.data(), s.meta.get(),
              (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(packedOut.data(), s.packed.get(), s.packedBytes);
}

void AfCodec::decode(dysco::span<const std::uint8_t> packedIn,
                     dysco::span<const float> metaIn,
                     dysco::span<const std::int32_t> ant1,
                     dysco::span<const std::int32_t> ant2,
                     dysco::span<float> interleavedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;
  if (packedIn.size() > s.packedBytes)
    throw std::runtime_error("Dysco GPU decode: packed buffer too small");

  std::memcpy(s.packed.get(), packedIn.data(), packedIn.size());
  std::memcpy(s.meta.get(), metaIn.data(),
              (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(s.ant1.get(), ant1.data(), s.nRows * sizeof(std::int32_t));
  std::memcpy(s.ant2.get(), ant2.data(), s.nRows * sizeof(std::int32_t));

  d1_bit_unpack(cuda::std::span<const std::uint8_t>(s.packed.get(),
                                                    packedIn.size()),
                s.bits,
                cuda::std::span<std::uint32_t>(s.symbols.get(), 2 * s.nVis),
                stream);
  d2_dequantize(s.symbolsMatrix(),
                cuda::std::span<const float>(s.meta.get(), s.visPerRow + s.nPol * s.nAnt),
                cuda::std::span<const std::int32_t>(s.ant1.get(), s.nRows),
                cuda::std::span<const std::int32_t>(s.ant2.get(), s.nRows), s.nRows, s.nChannels,
                s.nPol, s.nAnt,
                cuda::std::span<const float>(s.decValues.get(), s.decCount),
                s.outMatrix(), stream);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(interleavedOut.data(), s.out.get(), 2 * s.nVis * sizeof(float));
}

void quantize(dysco::span<const float> values,
              dysco::span<const float> decValues, std::uint64_t seed,
              std::uint64_t blockIndex, dysco::span<std::uint32_t> symbols) {
  const std::size_t count = values.size();
  if (symbols.size() < count)
    throw std::runtime_error("Dysco GPU quantize: symbol buffer too small");
  auto dValues = allocateManaged<float>(count);
  auto dDecValues = allocateManaged<float>(decValues.size());
  auto dSymbols = allocateManaged<std::uint32_t>(count);
  std::memcpy(dValues.get(), values.data(), count * sizeof(float));
  std::memcpy(dDecValues.get(), decValues.data(),
              decValues.size() * sizeof(float));
  k7_quantize_flat(cuda::std::span<const float>(dValues.get(), count),
                   cuda::std::span<const float>(dDecValues.get(),
                                                decValues.size()),
                   static_cast<std::uint32_t>(seed),
                   static_cast<std::uint32_t>(seed >> 32), blockIndex,
                   cuda::std::span<std::uint32_t>(dSymbols.get(), count),
                   DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(symbols.data(), dSymbols.get(),
              count * sizeof(std::uint32_t));
}

void pack(dysco::span<const std::uint32_t> symbols, std::uint32_t bits,
          dysco::span<std::uint8_t> packed) {
  auto dSymbols = allocateManaged<std::uint32_t>(symbols.size());
  auto dPacked = allocateManaged<std::uint8_t>(packed.size());
  std::memcpy(dSymbols.get(), symbols.data(),
              symbols.size() * sizeof(std::uint32_t));
  k8_bit_pack(
      cuda::std::span<const std::uint32_t>(dSymbols.get(), symbols.size()), bits,
      cuda::std::span<std::uint8_t>(dPacked.get(), packed.size()),
      DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(packed.data(), dPacked.get(), packed.size());
}

void unpack(dysco::span<const std::uint8_t> packed, std::uint32_t bits,
            dysco::span<std::uint32_t> symbols) {
  auto dPacked = allocateManaged<std::uint8_t>(packed.size());
  auto dSymbols = allocateManaged<std::uint32_t>(symbols.size());
  std::memcpy(dPacked.get(), packed.data(), packed.size());
  d1_bit_unpack(
      cuda::std::span<const std::uint8_t>(dPacked.get(), packed.size()), bits,
      cuda::std::span<std::uint32_t>(dSymbols.get(), symbols.size()),
      DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(symbols.data(), dSymbols.get(),
              symbols.size() * sizeof(std::uint32_t));
}

}  // namespace dyscostman::gpu
