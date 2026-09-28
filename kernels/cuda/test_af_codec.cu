// Standalone sanity / bit-compatibility test for the GPU AF codec.
// Build (CUDA only): see kernels/cuda/build_test.sh

#include "gpu_af_codec.h"

#include "../af_reference.h"
#include "../../bytepacker.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace dyscostman;
using namespace dyscostman::gpu;

static bool compareBytes(const void *a, const void *b, size_t n,
                         const char *what) {
  const unsigned char *pa = static_cast<const unsigned char *>(a);
  const unsigned char *pb = static_cast<const unsigned char *>(b);
  for (size_t i = 0; i != n; ++i) {
    if (pa[i] != pb[i]) {
      std::printf("  MISMATCH in %s at byte %zu: %02x vs %02x\n", what, i,
                  pa[i], pb[i]);
      return false;
    }
  }
  return true;
}

static int runCase(size_t nAnt, size_t nChannels, size_t nPol, unsigned bits,
                   bool fitToMaximum) {
  const size_t nRows = nAnt * (nAnt + 1) / 2;
  const size_t visPerRow = nChannels * nPol;
  const size_t nVis = nRows * visPerRow;
  const size_t decCount = (1u << bits) - 1;

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

  std::mt19937 gen(1234);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> input(2 * nVis);
  for (size_t i = 0; i != nVis; ++i) {
    const size_t row = i / visPerRow;
    const size_t pair = row;
    float gain = 1.0f + 0.5f * ((pair * 7) % 5);
    float val = gauss(gen) * gain;
    // sprinkle large outliers
    if (i % 97 == 0) val *= 50.0f;
    input[i * 2] = val;
    input[i * 2 + 1] = gauss(gen) * gain;
  }
  // A few NaNs, like flagged data
  for (size_t i = 0; i < nVis; i += 311) {
    input[i * 2] = std::numeric_limits<float>::quiet_NaN();
    input[i * 2 + 1] = std::numeric_limits<float>::quiet_NaN();
  }

  const size_t metaCount = visPerRow + nPol * nAnt;
  const uint64_t seed = 0x1122334455667788ull;
  const uint64_t blockIndex = 3;

  std::vector<float> metaCpu(metaCount);
  std::vector<unsigned int> symCpu(2 * nVis);
  cpuAfEncode(input, ant1, ant2, nChannels, nPol, nAnt, decValues, seed,
              blockIndex, fitToMaximum, metaCpu, symCpu);

  AfCodec codec(nRows, nChannels, nPol, nAnt, decValues, bits);
  std::vector<float> metaGpu(metaCount);
  std::vector<unsigned char> packedGpu(codec.packedByteCount());
  codec.encode(input, ant1, ant2, seed, blockIndex, fitToMaximum, metaGpu,
               packedGpu);

  bool ok = true;
  ok &= compareBytes(metaCpu.data(), metaGpu.data(),
                     metaCount * sizeof(float), "metadata");

  // CPU packed output via the production BytePacker
  std::vector<unsigned char> packedCpu(codec.packedByteCount());
  BytePacker::pack(bits, packedCpu.data(), symCpu.data(), 2 * nVis);
  ok &= compareBytes(packedCpu.data(), packedGpu.data(), packedCpu.size(),
                     "packed symbols");

  // Unpack GPU symbols and compare with the CPU symbols
  std::vector<unsigned int> symGpu(2 * nVis);
  BytePacker::unpack(bits, symGpu.data(), packedGpu.data(), 2 * nVis);
  ok &= compareBytes(symCpu.data(), symGpu.data(),
                     symCpu.size() * sizeof(unsigned int), "symbols");

  // Decode both and compare
  std::vector<float> outCpu(2 * nVis), outGpu(2 * nVis);
  cpuAfDecode(symCpu, metaCpu, ant1, ant2, nChannels, nPol, nAnt,
              decValues, outCpu);
  codec.decode(packedGpu, metaGpu, ant1, ant2, outGpu);
  ok &= compareBytes(outCpu.data(), outGpu.data(), outCpu.size() * sizeof(float),
                     "decoded data");

  std::printf("  nAnt=%zu C=%zu P=%zu bits=%u fit=%d : %s\n", nAnt, nChannels,
              nPol, bits, fitToMaximum ? 1 : 0, ok ? "OK" : "FAIL");
  return ok ? 0 : 1;
}

int main() {
  if (!available()) {
    std::printf("No GPU available\n");
    return 2;
  }
  std::printf("GPU: %.*s\n", static_cast<int>(deviceName().size()), deviceName().data());
  int failures = 0;
  failures += runCase(8, 16, 4, 8, false);
  failures += runCase(8, 16, 4, 8, true);
  failures += runCase(16, 8, 2, 4, false);
  failures += runCase(4, 8, 4, 12, true);
  std::printf("%s\n", failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED");
  return failures;
}
