// Benchmark: CPU reference (Philox) vs GPU AF encoder/decoder.
//
// The block under test is either the built-in synthetic MWA-like block
// (128 antennas, 8256 baselines, 96 channels, 4 polarizations, 8 bits) or the
// first time block of a real Measurement Set passed on the command line.
//
// Reported for both with and without `fitToMaximum`:
//
//   * the CPU reference encoder with the counter-based Philox dither
//     (`cpuAfEncode`, bit-identical to the GPU),
//   * the CUDA encode/decode codec.
//
// Usage: dsbench [iterations] [measurement-set]

#include "gpu_af_codec.h"

#include "../af_reference.h"

#include <chrono>
#include <fstream>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace dyscostman;
using namespace dyscostman::gpu;

namespace {

double nowSeconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

/// A single time block in the arrays the encoders consume.
struct BlockData {
  std::size_t nAnt = 0;
  std::size_t nChannels = 0;
  std::size_t nPol = 0;
  std::size_t nRows = 0;
  std::vector<std::int32_t> ant1, ant2;
  std::vector<float> input;  // 2 * nRows * nChannels * nPol, re/im interleaved
  std::string source;
};

BlockData makeSyntheticBlock() {
  BlockData block;
  block.nAnt = 128;
  block.nChannels = 96;
  block.nPol = 4;
  block.nRows = block.nAnt * (block.nAnt + 1) / 2;
  block.source = "synthetic MWA-like";
  const std::size_t visPerRow = block.nChannels * block.nPol;
  const std::size_t nVis = block.nRows * visPerRow;

  block.ant1.resize(block.nRows);
  block.ant2.resize(block.nRows);
  std::size_t r = 0;
  for (std::size_t a = 0; a != block.nAnt; ++a)
    for (std::size_t b = a; b != block.nAnt; ++b) {
      block.ant1[r] = static_cast<std::int32_t>(a);
      block.ant2[r] = static_cast<std::int32_t>(b);
      ++r;
    }

  std::mt19937 gen(42);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  block.input.resize(2 * nVis);
  for (std::size_t i = 0; i != nVis; ++i) {
    block.input[i * 2] = gauss(gen);
    block.input[i * 2 + 1] = gauss(gen);
  }
  return block;
}

/// Read the flat block produced by `tools/msblockextract.cc`.
BlockData loadFlatBlock(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open block file: " + path);

  auto readPod = [&](auto &value) {
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated block file: " + path);
  };

  char magic[8];
  in.read(magic, 8);
  if (in.gcount() != 8 || std::memcmp(magic, "DYSCOBK1", 8) != 0)
    throw std::runtime_error("bad magic in block file: " + path);

  BlockData block;
  std::uint64_t nAnt = 0, nChannels = 0, nPol = 0, nRows = 0;
  readPod(nAnt);
  readPod(nChannels);
  readPod(nPol);
  readPod(nRows);
  block.nAnt = static_cast<std::size_t>(nAnt);
  block.nChannels = static_cast<std::size_t>(nChannels);
  block.nPol = static_cast<std::size_t>(nPol);
  block.nRows = static_cast<std::size_t>(nRows);
  block.source = path;

  block.ant1.resize(block.nRows);
  block.ant2.resize(block.nRows);
  in.read(reinterpret_cast<char *>(block.ant1.data()),
          static_cast<std::streamsize>(block.nRows * sizeof(int)));
  in.read(reinterpret_cast<char *>(block.ant2.data()),
          static_cast<std::streamsize>(block.nRows * sizeof(int)));
  if (!in) throw std::runtime_error("truncated block file: " + path);

  const std::size_t nVis = block.nRows * block.nChannels * block.nPol;
  block.input.resize(2 * nVis);
  in.read(reinterpret_cast<char *>(block.input.data()),
          static_cast<std::streamsize>(block.input.size() * sizeof(float)));
  if (!in) throw std::runtime_error("truncated block file: " + path);
  return block;
}

}  // namespace

int main(int argc, char **argv) {
  const std::size_t iterations = argc > 1 ? std::atoi(argv[1]) : 3;

  if (!available()) {
    std::printf("No GPU available\n");
    return 2;
  }

  BlockData block;
  try {
    block = argc > 2 ? loadFlatBlock(argv[2]) : makeSyntheticBlock();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "Failed to load block: %s\n", e.what());
    return 1;
  }

  const unsigned bits = 8;
  const std::size_t visPerRow = block.nChannels * block.nPol;
  const std::size_t nVis = block.nRows * visPerRow;
  const std::size_t decCount = (1u << bits) - 1;
  const std::size_t nAnt = block.nAnt;
  const std::size_t nRows = block.nRows;
  const std::size_t nChannels = block.nChannels;
  const std::size_t nPol = block.nPol;

  std::printf("GPU: %.*s\n", static_cast<int>(deviceName().size()),
              deviceName().data());
  std::printf("source: %s\n", block.source.c_str());
  std::printf(
      "block: %zu rows x %zu ch x %zu pol, nAnt=%zu, bits=%u, nVis=%zu "
      "(%.1f MiB raw)\n",
      nRows, nChannels, nPol, nAnt, bits, nVis,
      (double)(nVis * 8) / 1024 / 1024);

  std::vector<float> decValues(decCount);
  for (std::size_t i = 0; i != decCount; ++i)
    decValues[i] = static_cast<float>(-2.5 + 5.0 * (i + 0.5) / decCount);

  const std::uint64_t seed = 0x1122334455667788ull;
  const std::size_t metaCount = visPerRow + nPol * nAnt;
  std::vector<float> metaCpu(metaCount), metaGpu(metaCount);
  std::vector<std::uint32_t> symCpu(2 * nVis), symGpu(2 * nVis);

  AfCodec codec(nRows, nChannels, nPol, nAnt, decValues, bits);
  std::vector<std::uint8_t> packedGpu(codec.packedByteCount());
  std::vector<float> outCpu(2 * nVis), outGpu(2 * nVis);

  // ---- warmup ----
  cpuAfEncode(block.input, block.ant1, block.ant2, nChannels, nPol, nAnt,
              decValues, seed, 0, false, metaCpu, symCpu);
  codec.encode(block.input, block.ant1, block.ant2, seed, 0, false, metaGpu,
               packedGpu);
  cpuAfDecode(symCpu, metaCpu, block.ant1, block.ant2, nChannels, nPol, nAnt,
              decValues, outCpu);
  codec.decode(packedGpu, metaGpu, block.ant1, block.ant2, outGpu);

  // ---- CPU encode (Philox reference) ----
  double t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    cpuAfEncode(block.input, block.ant1, block.ant2, nChannels, nPol, nAnt,
                decValues, seed, it, false, metaCpu, symCpu);
  const double tCpuPhi = (nowSeconds() - t0) / iterations;

  t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    cpuAfEncode(block.input, block.ant1, block.ant2, nChannels, nPol, nAnt,
                decValues, seed, it, true, metaCpu, symCpu);
  const double tCpuPhiFit = (nowSeconds() - t0) / iterations;

  // ---- GPU encode ----
  t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    codec.encode(block.input, block.ant1, block.ant2, seed, it, false, metaGpu,
                 packedGpu);
  const double tGpu = (nowSeconds() - t0) / iterations;

  t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    codec.encode(block.input, block.ant1, block.ant2, seed, it, true, metaGpu,
                 packedGpu);
  const double tGpuFit = (nowSeconds() - t0) / iterations;

  // ---- decode ----
  t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    cpuAfDecode(symCpu, metaCpu, block.ant1, block.ant2, nChannels, nPol, nAnt,
                decValues, outCpu);
  const double tCpuDec = (nowSeconds() - t0) / iterations;

  t0 = nowSeconds();
  for (std::size_t it = 0; it != iterations; ++it)
    codec.decode(packedGpu, metaGpu, block.ant1, block.ant2, outGpu);
  const double tGpuDec = (nowSeconds() - t0) / iterations;

  const double rawMiB = static_cast<double>(nVis * 8) / 1024 / 1024;
  auto rate = [&](double sec) { return rawMiB / sec; };

  std::printf("\n%-32s %10s %14s\n", "stage", "time [s]", "throughput");
  std::printf("%-32s %10.4f %11.1f MB/s\n", "CPU encode (Philox, no fit)",
              tCpuPhi, rate(tCpuPhi));
  std::printf("%-32s %10.4f %11.1f MB/s\n", "CPU encode (Philox, fit-to-max)",
              tCpuPhiFit, rate(tCpuPhiFit));
  std::printf("%-32s %10.4f %11.1f MB/s  (vs CPU %.1fx)\n",
              "GPU encode (no fit)", tGpu, rate(tGpu), tCpuPhi / tGpu);
  std::printf("%-32s %10.4f %11.1f MB/s  (vs CPU-fit %.1fx)\n",
              "GPU encode (fit-to-max)", tGpuFit, rate(tGpuFit),
              tCpuPhiFit / tGpuFit);
  std::printf("%-32s %10.4f %11.1f MB/s\n", "CPU decode", tCpuDec,
              rate(tCpuDec));
  std::printf("%-32s %10.4f %11.1f MB/s  (speedup %.1fx)\n", "GPU decode",
              tGpuDec, rate(tGpuDec), tCpuDec / tGpuDec);
  return 0;
}
