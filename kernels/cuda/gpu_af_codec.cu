// Orchestration of the AF GPU codec: workspace management (CUDA managed
// memory / transparent page migration) and the fixed pipeline of kernel
// launches.  No computation happens here; this file only wires the individual
// per-kernel files together.

#include "gpu_af_codec.h"

#include "cuda_kernels.h"
#include "../gpu_portability.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace dyscostman::gpu {

bool available() {
  int count = 0;
  const cudaError_t error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess)
    std::fprintf(stderr, "[dysco-gpu] cudaGetDeviceCount: %s\n",
                 cudaGetErrorString(error));
  return error == cudaSuccess && count > 0;
}

std::string_view deviceName() { return dysco_gpu::device_name(); }

namespace {
[[nodiscard]] constexpr std::size_t packedByteCountFor(
    std::size_t symbolCount, unsigned bits) noexcept {
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
  const unsigned bits;
  const std::size_t packedBytes;

  Managed<float> decValues;
  Managed<float> inInterleaved;  // 2*nVis
  Managed<double> re, im;
  Managed<int> ant1, ant2;
  Managed<float> meta;  // visPerRow + nPol*nAnt
  Managed<double> matrix;
  Managed<double> rmsPerAnt;
  Managed<unsigned int> symbols;
  Managed<unsigned char> packed;
  Managed<float> out;

  Impl(std::size_t rows, std::size_t channels, std::size_t pol,
       std::size_t antennas, dysco::span<const float> decValuesIn,
       unsigned bitsPerSymbol)
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
        ant1(allocate<int>(nRows)),
        ant2(allocate<int>(nRows)),
        meta(allocate<float>(visPerRow + nPol * nAnt)),
        matrix(allocate<double>(nAnt * nAnt)),
        rmsPerAnt(allocate<double>(nAnt)),
        symbols(allocate<unsigned int>(2 * nVis)),
        packed(allocate<unsigned char>(packedBytes)),
        out(allocate<float>(2 * nVis)) {
    std::copy(decValuesIn.begin(), decValuesIn.end(), decValues.get());
  }

  [[nodiscard]] double maxLevel() const noexcept {
    return decCount == 0 ? 0.0 : static_cast<double>(decValues.get()[decCount - 1]);
  }
};

AfCodec::AfCodec(std::size_t nRows, std::size_t nChannels, std::size_t nPol,
                 std::size_t nAnt, dysco::span<const float> decValues,
                 unsigned bitsPerSymbol)
    : _impl(std::make_unique<Impl>(nRows, nChannels, nPol, nAnt, decValues,
                                   bitsPerSymbol)) {}

AfCodec::~AfCodec() = default;

std::size_t AfCodec::packedByteCount() const noexcept {
  return _impl->packedBytes;
}

void AfCodec::encode(dysco::span<const float> interleavedIn,
                     dysco::span<const int> ant1, dysco::span<const int> ant2,
                     std::uint64_t seed, std::uint64_t blockIndex,
                     bool fitToMaximum, dysco::span<float> metaOut,
                     dysco::span<unsigned char> packedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;

  // Upload the block (transparent page migration: a plain CPU write to the
  // managed buffer is enough; pages move on first device access).
  std::memcpy(s.inInterleaved.get(), interleavedIn.data(),
              2 * s.nVis * sizeof(float));
  std::memcpy(s.ant1.get(), ant1.data(), s.nRows * sizeof(int));
  std::memcpy(s.ant2.get(), ant2.data(), s.nRows * sizeof(int));

  k1_gather_convert({s.inInterleaved.get(), 2 * s.nVis}, s.nRows, s.visPerRow,
                    {s.re.get(), s.nVis}, {s.im.get(), s.nVis}, stream);

  k2_channel_rms({s.re.get(), s.nVis}, {s.im.get(), s.nVis}, s.nRows,
                 s.visPerRow, {s.meta.get(), s.visPerRow}, stream);
  k5a_apply_channel_norm({s.re.get(), s.nVis}, {s.im.get(), s.nVis},
                         s.visPerRow, {s.meta.get(), s.visPerRow}, stream);

  for (std::size_t pol = 0; pol != s.nPol; ++pol) {
    k3_antenna_rms_matrix({s.re.get(), s.nVis}, {s.im.get(), s.nVis},
                          {s.ant1.get(), s.nRows}, {s.ant2.get(), s.nRows},
                          s.nRows, s.nChannels, s.nPol, s.nAnt, pol,
                          {s.matrix.get(), s.nAnt * s.nAnt}, stream);
    k4_antenna_gain_solve({s.matrix.get(), s.nAnt * s.nAnt}, s.nAnt,
                          {s.rmsPerAnt.get(), s.nAnt}, stream);
    k5b_apply_antenna_norm({s.re.get(), s.nVis}, {s.im.get(), s.nVis},
                           {s.ant1.get(), s.nRows}, {s.ant2.get(), s.nRows},
                           s.nRows, s.nChannels, s.nPol, pol,
                           {s.rmsPerAnt.get(), s.nAnt}, stream);
    // Store the antenna gains in the metadata block, in the same order as the
    // CPU encoder ([visPerRow + pol*nAnt + antenna]).
    k4_store_antenna_gains({s.rmsPerAnt.get(), s.nAnt},
                           {s.meta.get() + s.visPerRow + pol * s.nAnt, s.nAnt},
                           stream);
  }

  if (fitToMaximum) {
    k6_fit_to_max({s.re.get(), s.nVis}, {s.im.get(), s.nVis},
                  {s.ant1.get(), s.nRows}, {s.ant2.get(), s.nRows}, s.nRows,
                  s.nChannels, s.nPol, s.nAnt,
                  {s.meta.get(), s.visPerRow + s.nPol * s.nAnt}, s.maxLevel(),
                  stream);
  }

  k7_quantize_dither(
      {s.re.get(), s.nVis}, {s.im.get(), s.nVis},
      {s.decValues.get(), s.decCount}, static_cast<std::uint32_t>(seed),
      static_cast<std::uint32_t>(seed >> 32), blockIndex,
      {s.symbols.get(), 2 * s.nVis}, stream);
  k8_bit_pack({s.symbols.get(), 2 * s.nVis}, s.bits,
              {s.packed.get(), s.packedBytes}, stream);

  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(metaOut.data(), s.meta.get(),
              (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(packedOut.data(), s.packed.get(), s.packedBytes);
}

void AfCodec::decode(dysco::span<const unsigned char> packedIn,
                     dysco::span<const float> metaIn,
                     dysco::span<const int> ant1, dysco::span<const int> ant2,
                     dysco::span<float> interleavedOut) {
  Impl &s = *_impl;
  const dysco_stream_t stream = DYSCO_DEFAULT_STREAM;
  if (packedIn.size() > s.packedBytes)
    throw std::runtime_error("Dysco GPU decode: packed buffer too small");

  std::memcpy(s.packed.get(), packedIn.data(), packedIn.size());
  std::memcpy(s.meta.get(), metaIn.data(),
              (s.visPerRow + s.nPol * s.nAnt) * sizeof(float));
  std::memcpy(s.ant1.get(), ant1.data(), s.nRows * sizeof(int));
  std::memcpy(s.ant2.get(), ant2.data(), s.nRows * sizeof(int));

  d1_bit_unpack({s.packed.get(), packedIn.size()}, s.bits,
                {s.symbols.get(), 2 * s.nVis}, stream);
  d2_dequantize({s.symbols.get(), 2 * s.nVis}, s.nRows, s.nChannels, s.nPol,
                s.nAnt, {s.meta.get(), s.visPerRow + s.nPol * s.nAnt},
                {s.ant1.get(), s.nRows}, {s.ant2.get(), s.nRows},
                {s.decValues.get(), s.decCount}, {s.out.get(), 2 * s.nVis},
                stream);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(interleavedOut.data(), s.out.get(), 2 * s.nVis * sizeof(float));
}

void quantize(dysco::span<const float> values,
              dysco::span<const float> decValues, std::uint64_t seed,
              std::uint64_t blockIndex, dysco::span<unsigned int> symbols) {
  const std::size_t count = values.size();
  if (symbols.size() < count)
    throw std::runtime_error("Dysco GPU quantize: symbol buffer too small");
  auto dValues = allocateManaged<float>(count);
  auto dDecValues = allocateManaged<float>(decValues.size());
  auto dSymbols = allocateManaged<unsigned int>(count);
  std::memcpy(dValues.get(), values.data(), count * sizeof(float));
  std::memcpy(dDecValues.get(), decValues.data(),
              decValues.size() * sizeof(float));
  k7_quantize_flat({dValues.get(), count}, {dDecValues.get(), decValues.size()},
                   static_cast<std::uint32_t>(seed),
                   static_cast<std::uint32_t>(seed >> 32), blockIndex,
                   {dSymbols.get(), count}, DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(symbols.data(), dSymbols.get(), count * sizeof(unsigned int));
}

void pack(dysco::span<const unsigned int> symbols, unsigned bits,
          dysco::span<unsigned char> packed) {
  auto dSymbols = allocateManaged<unsigned int>(symbols.size());
  auto dPacked = allocateManaged<unsigned char>(packed.size());
  std::memcpy(dSymbols.get(), symbols.data(),
              symbols.size() * sizeof(unsigned int));
  k8_bit_pack({dSymbols.get(), symbols.size()}, bits,
              {dPacked.get(), packed.size()}, DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(packed.data(), dPacked.get(), packed.size());
}

void unpack(dysco::span<const unsigned char> packed, unsigned bits,
            dysco::span<unsigned int> symbols) {
  auto dPacked = allocateManaged<unsigned char>(packed.size());
  auto dSymbols = allocateManaged<unsigned int>(symbols.size());
  std::memcpy(dPacked.get(), packed.data(), packed.size());
  d1_bit_unpack({dPacked.get(), packed.size()}, bits,
                {dSymbols.get(), symbols.size()}, DYSCO_DEFAULT_STREAM);
  DYSCO_GPU_CHECK(cudaDeviceSynchronize());
  std::memcpy(symbols.data(), dSymbols.get(),
              symbols.size() * sizeof(unsigned int));
}

}  // namespace dyscostman::gpu
