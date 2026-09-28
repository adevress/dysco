// Benchmark: CPU reference AF encoder/decoder vs GPU codec.
//
// Uses an MWA-like time block (128 antennas, 8256 baselines, 96 channels,
// 4 polarizations, 8 bits) and reports wall-clock time and throughput.

#include "gpu_af_codec.h"

#include "../af_reference.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace dyscostman;
using namespace dyscostman::gpu;

static double nowSeconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  const size_t nAnt = 128;
  const size_t nChannels = 96;
  const size_t nPol = 4;
  const unsigned bits = 8;
  const size_t nRows = nAnt * (nAnt + 1) / 2;
  const size_t visPerRow = nChannels * nPol;
  const size_t nVis = nRows * visPerRow;
  const size_t decCount = (1u << bits) - 1;
  const size_t iterations = argc > 1 ? std::atoi(argv[1]) : 3;

  if (!available()) {
    std::printf("No GPU available\n");
    return 2;
  }
  std::printf("GPU: %s\n", deviceName());
  std::printf("block: %zu rows x %zu ch x %zu pol, bits=%u, nVis=%zu (%.1f MiB raw)\n",
              nRows, nChannels, nPol, bits, nVis,
              (double)(nVis * 8) / 1024 / 1024);

  std::vector<float> decValues(decCount);
  for (size_t i = 0; i != decCount; ++i)
    decValues[i] = static_cast<float>(-2.5 + 5.0 * (i + 0.5) / decCount);

  std::vector<int> ant1(nRows), ant2(nRows);
  size_t r = 0;
  for (size_t a = 0; a != nAnt; ++a)
    for (size_t b = a; b != nAnt; ++b) {
      ant1[r] = static_cast<int>(a);
      ant2[r] = static_cast<int>(b);
      ++r;
    }

  std::mt19937 gen(42);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> input(2 * nVis);
  for (size_t i = 0; i != nVis; ++i) {
    input[i * 2] = gauss(gen);
    input[i * 2 + 1] = gauss(gen);
  }

  const uint64_t seed = 0x1122334455667788ull;
  const size_t metaCount = visPerRow + nPol * nAnt;
  std::vector<float> metaCpu(metaCount), metaGpu(metaCount);
  std::vector<unsigned int> symCpu(2 * nVis), symGpu(2 * nVis);

  AfCodec codec(nRows, nChannels, nPol, nAnt, decValues.data(), decCount, bits);
  std::vector<unsigned char> packedGpu(codec.packedByteCount());
  std::vector<float> outCpu(2 * nVis), outGpu(2 * nVis);

  // ---- warmup ----
  cpuAfEncode(input.data(), ant1.data(), ant2.data(), nRows, nChannels, nPol,
              nAnt, decValues.data(), decCount, seed, 0, false, metaCpu.data(),
              symCpu.data());
  codec.encode(input.data(), ant1.data(), ant2.data(), seed, 0, false,
               metaGpu.data(), packedGpu.data());
  cpuAfDecode(symCpu.data(), metaCpu.data(), ant1.data(), ant2.data(), nRows,
              nChannels, nPol, nAnt, decValues.data(), decCount, outCpu.data());
  codec.decode(packedGpu.data(), packedGpu.size(), metaGpu.data(), ant1.data(),
               ant2.data(), outGpu.data());

  // ---- CPU encode ----
  double t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    cpuAfEncode(input.data(), ant1.data(), ant2.data(), nRows, nChannels, nPol,
                nAnt, decValues.data(), decCount, seed, it, false,
                metaCpu.data(), symCpu.data());
  double tCpuEnc = (nowSeconds() - t0) / iterations;

  // ---- CPU encode (with fit) ----
  t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    cpuAfEncode(input.data(), ant1.data(), ant2.data(), nRows, nChannels, nPol,
                nAnt, decValues.data(), decCount, seed, it, true,
                metaCpu.data(), symCpu.data());
  double tCpuEncFit = (nowSeconds() - t0) / iterations;

  // ---- GPU encode (no fit) ----
  t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    codec.encode(input.data(), ant1.data(), ant2.data(), seed, it, false,
                 metaGpu.data(), packedGpu.data());
  double tGpuEnc = (nowSeconds() - t0) / iterations;

  // ---- GPU encode (with fit-to-maximum) ----
  t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    codec.encode(input.data(), ant1.data(), ant2.data(), seed, it, true,
                 metaGpu.data(), packedGpu.data());
  double tGpuEncFit = (nowSeconds() - t0) / iterations;

  // ---- CPU decode ----
  t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    cpuAfDecode(symCpu.data(), metaCpu.data(), ant1.data(), ant2.data(), nRows,
                nChannels, nPol, nAnt, decValues.data(), decCount,
                outCpu.data());
  double tCpuDec = (nowSeconds() - t0) / iterations;

  // ---- GPU decode ----
  t0 = nowSeconds();
  for (size_t it = 0; it != iterations; ++it)
    codec.decode(packedGpu.data(), packedGpu.size(), metaGpu.data(),
                 ant1.data(), ant2.data(), outGpu.data());
  double tGpuDec = (nowSeconds() - t0) / iterations;

  const double rawMiB = static_cast<double>(nVis * 8) / 1024 / 1024;
  auto rate = [&](double sec) { return rawMiB / sec; };

  std::printf("\n%-28s %10s %12s\n", "stage", "time [s]", "throughput");
  std::printf("%-28s %10.4f %9.1f MB/s\n", "CPU encode (no fit)", tCpuEnc,
              rate(tCpuEnc));
  std::printf("%-28s %10.4f %9.1f MB/s  (speedup %.1fx)\n",
              "GPU encode (no fit)", tGpuEnc, rate(tGpuEnc),
              tCpuEnc / tGpuEnc);
  std::printf("%-28s %10.4f %9.1f MB/s  (speedup %.1fx)\n",
              "CPU encode (fit-to-max)", tCpuEncFit, rate(tCpuEncFit),
              tCpuEnc / tCpuEncFit);
  std::printf("%-28s %10.4f %9.1f MB/s  (vs CPU-fit %.1fx)\n",
              "GPU encode (fit-to-max)", tGpuEncFit, rate(tGpuEncFit),
              tCpuEncFit / tGpuEncFit);
  std::printf("%-28s %10.4f %9.1f MB/s\n", "CPU decode", tCpuDec,
              rate(tCpuDec));
  std::printf("%-28s %10.4f %9.1f MB/s  (speedup %.1fx)\n", "GPU decode",
              tGpuDec, rate(tGpuDec), tCpuDec / tGpuDec);
  return 0;
}
