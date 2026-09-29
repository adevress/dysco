// Standalone sanity / bit-compatibility test for the GPU AF codec.
// Build (CUDA only): see kernels/cuda/build_test.sh

#include "gpu_af_codec.h"

#include "../af_reference.h"
#include "../../bytepacker.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <type_traits>
#include <vector>

// Boundary assertion: the production BytePacker (the storage format) is typed
// in terms of `unsigned int`, while the GPU codec uses the fixed-width alias.
static_assert(std::is_same_v<std::uint32_t, unsigned int>);

using namespace dyscostman;
using namespace dyscostman::gpu;

static bool compareBytes(const void *a, const void *b, std::size_t n,
                         const char *what) {
  const unsigned char *pa = static_cast<const unsigned char *>(a);
  const unsigned char *pb = static_cast<const unsigned char *>(b);
  for (std::size_t i = 0; i != n; ++i) {
    if (pa[i] != pb[i]) {
      std::printf("  MISMATCH in %s at byte %zu: %02x vs %02x\n", what, i,
                  pa[i], pb[i]);
      return false;
    }
  }
  return true;
}

static int runCase(std::size_t nAnt, std::size_t nChannels, std::size_t nPol, unsigned bits,
                   bool fitToMaximum) {
  const std::size_t nRows = nAnt * (nAnt + 1) / 2;
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t nVis = nRows * visPerRow;
  const std::size_t decCount = (1u << bits) - 1;

  std::vector<float> decValues(decCount);
  for (std::size_t i = 0; i != decCount; ++i)
    decValues[i] = static_cast<float>(-2.5 + 5.0 * (i + 0.5) / decCount);

  std::vector<std::int32_t> ant1(nRows), ant2(nRows);
  std::size_t r = 0;
  for (std::size_t a = 0; a != nAnt; ++a)
    for (std::size_t b = a; b != nAnt; ++b) {
      ant1[r] = static_cast<std::int32_t>(a);
      ant2[r] = static_cast<std::int32_t>(b);
      ++r;
    }

  std::mt19937 gen(1234);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  std::vector<float> input(2 * nVis);
  for (std::size_t i = 0; i != nVis; ++i) {
    const std::size_t row = i / visPerRow;
    const std::size_t pair = row;
    float gain = 1.0f + 0.5f * ((pair * 7) % 5);
    float val = gauss(gen) * gain;
    // sprinkle large outliers
    if (i % 97 == 0) val *= 50.0f;
    input[i * 2] = val;
    input[i * 2 + 1] = gauss(gen) * gain;
  }
  // A few NaNs, like flagged data
  for (std::size_t i = 0; i < nVis; i += 311) {
    input[i * 2] = std::numeric_limits<float>::quiet_NaN();
    input[i * 2 + 1] = std::numeric_limits<float>::quiet_NaN();
  }

  const std::size_t metaCount = visPerRow + nPol * nAnt;
  const std::uint64_t seed = 0x1122334455667788ull;
  const std::uint64_t blockIndex = 3;

  std::vector<float> metaCpu(metaCount);
  std::vector<std::uint32_t> symCpu(2 * nVis);
  cpuAfEncode(input, ant1, ant2, nChannels, nPol, nAnt, decValues, seed,
              blockIndex, fitToMaximum, metaCpu, symCpu);

  AfCodec codec(nRows, nChannels, nPol, nAnt, decValues, bits);
  std::vector<float> metaGpu(metaCount);
  std::vector<std::uint8_t> packedGpu(codec.packedByteCount());
  codec.encode(input, ant1, ant2, seed, blockIndex, fitToMaximum, metaGpu,
               packedGpu);

  bool ok = true;
  // Without fit-to-max the GPU path is bit-identical to the CPU reference.
  // With fit-to-max the default GPU algorithm is the parallel Jacobi variant,
  // which is only required to agree within a tolerance.
  if (!fitToMaximum) {
    ok &= compareBytes(metaCpu.data(), metaGpu.data(),
                       metaCount * sizeof(float), "metadata");

    // CPU packed output via the production BytePacker
    std::vector<std::uint8_t> packedCpu(codec.packedByteCount());
    BytePacker::pack(bits, packedCpu.data(), symCpu.data(), 2 * nVis);
    ok &= compareBytes(packedCpu.data(), packedGpu.data(), packedCpu.size(),
                       "packed symbols");

    // Unpack GPU symbols and compare with the CPU symbols
    std::vector<std::uint32_t> symGpu(2 * nVis);
    BytePacker::unpack(bits, symGpu.data(), packedGpu.data(), 2 * nVis);
    ok &= compareBytes(symCpu.data(), symGpu.data(),
                       symCpu.size() * sizeof(std::uint32_t), "symbols");
  } else {
    for (std::size_t i = 0; i != metaCount; ++i) {
      const float a = metaCpu[i], b = metaGpu[i];
      const float tolerance = 1e-2f * std::max(1.0f, std::fabs(a));
      if (std::fabs(a - b) > tolerance) {
        std::printf("  MISMATCH in metadata at %zu: %g vs %g\n", i, a, b);
        ok = false;
        break;
      }
    }
  }

  // Decode both and compare
  std::vector<float> outCpu(2 * nVis), outGpu(2 * nVis);
  cpuAfDecode(symCpu, metaCpu, ant1, ant2, nChannels, nPol, nAnt,
              decValues, outCpu);
  codec.decode(packedGpu, metaGpu, ant1, ant2, outGpu);
  if (!fitToMaximum) {
    ok &= compareBytes(outCpu.data(), outGpu.data(),
                       outCpu.size() * sizeof(float), "decoded data");
  } else {
    double error = 0.0, signal = 0.0;
    for (std::size_t i = 0; i != outCpu.size(); ++i) {
      const double d = static_cast<double>(outGpu[i]) - outCpu[i];
      error += d * d;
      signal += static_cast<double>(outCpu[i]) * outCpu[i];
    }
    if (std::sqrt(error) > std::sqrt(signal) * 1e-2 + 1e-4) {
      std::printf("  MISMATCH decoded data: relative rms error %.4g\n",
                  std::sqrt(error) / (std::sqrt(signal) + 1e-30));
      ok = false;
    }
  }

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
