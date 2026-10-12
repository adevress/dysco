#include "af_reference.h"

#include "../preconditions.hpp"
#include "philox_rng.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace dyscostman::gpu {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// K7 helper: the CPU mirror of quantizeDither().
[[nodiscard]] std::uint32_t quantizeDitherRef(
    float value, dysco::span<const float> decValues,
    std::uint32_t dither) noexcept {
  const std::size_t decCount = decValues.size();
  if (!std::isfinite(value)) return static_cast<std::uint32_t>(decCount);
  std::size_t lo = 0, hi = decCount;
  while (lo < hi) {
    const std::size_t mid = (lo + hi) >> 1;
    if (decValues[mid] < value)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0) return 0;
  if (lo >= decCount) return static_cast<std::uint32_t>(decCount - 1);
  const float rightValue = decValues[lo];
  const float leftValue = decValues[lo - 1];
  const float ditherMark =
      static_cast<float>(2147483648.0f * (value - leftValue) /
                         (rightValue - leftValue));
  return ditherMark > static_cast<float>(dither)
             ? static_cast<std::uint32_t>(lo)
             : static_cast<std::uint32_t>(lo - 1);
}

/// K4 mirror.
void solveAntennaGains(dysco::span<const double> matrix, std::size_t nAnt,
                       dysco::span<double> rms) {
  std::fill(rms.begin(), rms.end(), 1.0);
  double precision = 1.0;
  std::vector<double> next(nAnt);
  for (std::size_t iteration = 0; iteration != 100 && precision > 1e-6;
       ++iteration) {
    for (std::size_t i = 0; i != nAnt; ++i) {
      double weightSum = 0.0, acc = 0.0;
      for (std::size_t j = 0; j != nAnt; ++j) {
        if (i != j && rms[j] != 0.0) {
          const double m = matrix[i * nAnt + j];
          if (std::isfinite(m)) {
            acc += m;
            weightSum += rms[j];
          }
        }
      }
      next[i] = weightSum == 0.0 ? 0.0 : acc / weightSum;
    }
    double maxVal = 0.0;
    for (std::size_t i = 0; i != nAnt; ++i) {
      rms[i] = next[i] * 0.8 + rms[i] * 0.2;
      // `std::max` on NaN is formally UB (NaN is not LessThanComparable);
      // the reference path never produces NaN here, so make that explicit.
      enforce_precondition(std::isfinite(rms[i]));
      maxVal = std::max(maxVal, rms[i]);
    }
    precision = 0.0;
    for (std::size_t i = 0; i != nAnt; ++i) {
      if (rms[i] < maxVal * 1e-5) rms[i] = 0.0;
      const double relativeChange = std::fabs(rms[i] - next[i]) / maxVal;
      enforce_precondition(std::isfinite(relativeChange));
      precision = std::max(precision, relativeChange);
    }
  }
}

/// K6 CPU mirror of `AFTimeBlockEncoder::fitToMaximum`: the legacy greedy
/// algorithm.  This is what the CPU encoder keeps using; the GPU backend uses
/// the parallel Jacobi variant (`kernels/cuda/k6_fit_to_max.cu`) instead.
void fitToMaximumRef(dysco::span<double> re, dysco::span<double> im,
                     dysco::span<const std::int32_t> ant1,
                     dysco::span<const std::int32_t> ant2,
                     std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                     dysco::span<float> meta, double maxLevel) {
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t nRows = ant1.size();

  const auto changeChannelFactor = [&](std::size_t visIndex, double factor) {
    meta[visIndex] =
        static_cast<float>(static_cast<double>(meta[visIndex]) / factor);
    for (std::size_t row = 0; row != nRows; ++row) {
      const std::size_t idx = row * visPerRow + visIndex;
      re[idx] *= factor;
      im[idx] *= factor;
    }
  };
  const auto changeAntennaFactor = [&](std::size_t antennaIndex,
                                       std::size_t polIndex, double factor) {
    const std::size_t metaIndex = visPerRow + nAnt * polIndex + antennaIndex;
    meta[metaIndex] =
        static_cast<float>(static_cast<double>(meta[metaIndex]) / factor);
    for (std::size_t row = 0; row != nRows; ++row) {
      const auto a1 = static_cast<std::size_t>(ant1[row]);
      const auto a2 = static_cast<std::size_t>(ant2[row]);
      const std::uint32_t repeat =
          (a1 == antennaIndex ? 1u : 0u) + (a2 == antennaIndex ? 1u : 0u);
      for (std::uint32_t r = 0; r != repeat; ++r)
        for (std::size_t ch = 0; ch != nChannels; ++ch) {
          const std::size_t idx = row * visPerRow + ch * nPol + polIndex;
          re[idx] *= factor;
          im[idx] *= factor;
        }
    }
  };

  for (std::size_t visIndex = 0; visIndex != visPerRow; ++visIndex) {
    double largest = 0.0;
    for (std::size_t row = 0; row != nRows; ++row) {
      if (ant1[row] != ant2[row]) {
        const std::size_t idx = row * visPerRow + visIndex;
        const double local =
            std::max(std::max(re[idx], im[idx]), -std::min(re[idx], im[idx]));
        if (std::isfinite(local) && local > largest) largest = local;
      }
    }
    changeChannelFactor(
        visIndex,
        (maxLevel == 0.0 || largest == 0.0) ? 1.0 : maxLevel / largest);
  }

  std::vector<double> largestCh(nChannels);
  std::vector<double> maxCompPerAntenna(nAnt);
  std::vector<double> increasePerAntenna(nAnt);
  for (std::size_t polIndex = 0; polIndex != nPol; ++polIndex) {
    bool isProgressing;
    do {
      double bestChannelIncrease = 0.0, channelFactor = 1.0;
      std::size_t bestChannel = 0;
      for (std::size_t channel = 0; channel != nChannels; ++channel) {
        const std::size_t visIndex = channel * nPol + polIndex;
        double largest = 0.0;
        for (std::size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const std::size_t idx = row * visPerRow + visIndex;
            const double local = std::max(std::max(re[idx], im[idx]),
                                          -std::min(re[idx], im[idx]));
            if (std::isfinite(local) && local > largest) largest = local;
          }
        }
        largestCh[channel] = largest;
        const double factor =
            largest == 0.0 ? 0.0 : (maxLevel / largest - 1.0);
        double thisIncrease = 0.0;
        for (std::size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const std::size_t idx = row * visPerRow + visIndex;
            const double av =
                std::fabs(re[idx] * factor) + std::fabs(im[idx] * factor);
            if (std::isfinite(av)) thisIncrease += av;
          }
        }
        if (thisIncrease > bestChannelIncrease) {
          bestChannelIncrease = thisIncrease;
          bestChannel = channel;
          channelFactor = factor + 1.0;
        }
      }

      std::fill(maxCompPerAntenna.begin(), maxCompPerAntenna.end(), 0.0);
      for (std::size_t row = 0; row != nRows; ++row) {
        if (ant1[row] == ant2[row]) continue;
        const auto a1 = static_cast<std::size_t>(ant1[row]);
        const auto a2 = static_cast<std::size_t>(ant2[row]);
        for (std::size_t channel = 0; channel != nChannels; ++channel) {
          const std::size_t idx = row * visPerRow + channel * nPol + polIndex;
          const double local = std::max(std::max(re[idx], im[idx]),
                                        -std::min(re[idx], im[idx]));
          if (std::isfinite(local)) {
            maxCompPerAntenna[a1] = std::max(maxCompPerAntenna[a1], local);
            maxCompPerAntenna[a2] = std::max(maxCompPerAntenna[a2], local);
          }
        }
      }
      std::fill(increasePerAntenna.begin(), increasePerAntenna.end(), 0.0);
      for (std::size_t row = 0; row != nRows; ++row) {
        if (ant1[row] == ant2[row]) continue;
        const auto a1 = static_cast<std::size_t>(ant1[row]);
        const auto a2 = static_cast<std::size_t>(ant2[row]);
        const double factor1 = maxCompPerAntenna[a1] == 0.0
                                   ? 0.0
                                   : (maxLevel / maxCompPerAntenna[a1] - 1.0);
        const double factor2 = maxCompPerAntenna[a2] == 0.0
                                   ? 0.0
                                   : (maxLevel / maxCompPerAntenna[a2] - 1.0);
        for (std::size_t channel = 0; channel != nChannels; ++channel) {
          const std::size_t idx = row * visPerRow + channel * nPol + polIndex;
          const double av1 =
              std::fabs(re[idx] * factor1) + std::fabs(im[idx] * factor1);
          if (std::isfinite(av1)) increasePerAntenna[a1] += av1;
          const double av2 =
              std::fabs(re[idx] * factor2) + std::fabs(im[idx] * factor2);
          if (std::isfinite(av2)) increasePerAntenna[a2] += av2;
        }
      }
      std::size_t bestAntenna = 0;
      double bestAntennaIncrease = 0.0;
      for (std::size_t a = 0; a != nAnt; ++a) {
        if (increasePerAntenna[a] > bestAntennaIncrease) {
          bestAntennaIncrease = increasePerAntenna[a];
          bestAntenna = a;
        }
      }
      if (bestAntennaIncrease > bestChannelIncrease) {
        const double factor = maxCompPerAntenna[bestAntenna] == 0.0
                                  ? 1.0
                                  : (maxLevel / maxCompPerAntenna[bestAntenna]);
        if (factor < 1.0)
          isProgressing = false;
        else {
          isProgressing = factor > 1.01;
          changeAntennaFactor(bestAntenna, polIndex, factor);
        }
      } else {
        if (channelFactor < 1.0)
          isProgressing = false;
        else {
          isProgressing = channelFactor > 1.001;
          changeChannelFactor(bestChannel * nPol + polIndex, channelFactor);
        }
      }
    } while (isProgressing);
  }
}

}  // namespace

void cpuAfEncode(dysco::span<const float> interleaved,
                 dysco::span<const std::int32_t> ant1,
                 dysco::span<const std::int32_t> ant2,
                 std::size_t nChannels, std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues, std::uint64_t seed,
                 std::uint64_t blockIndex, bool fitToMaximum,
                 dysco::span<float> metaOut,
                 dysco::span<std::uint32_t> symbolsOut) {
  const std::size_t nRows = ant1.size();
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t nVis = nRows * visPerRow;
  const std::size_t decCount = decValues.size();
  std::vector<double> re(nVis), im(nVis);

  // K1
  for (std::size_t i = 0; i != nVis; ++i) {
    re[i] = static_cast<double>(interleaved[i * 2]);
    im[i] = static_cast<double>(interleaved[i * 2 + 1]);
  }
  // K2
  for (std::size_t i = 0; i != visPerRow; ++i) {
    double value = 0.0;
    std::size_t count = 0;
    for (std::size_t row = 0; row != nRows; ++row) {
      const std::size_t idx = row * visPerRow + i;
      if (std::isfinite(re[idx]) && std::isfinite(im[idx])) {
        ++count;
        value += re[idx] * re[idx] + im[idx] * im[idx];
      }
    }
    metaOut[i] = static_cast<float>(std::sqrt(value / (count * 2.0)));
  }
  // K5a (guarded exactly like the production CPU encoder)
  for (std::size_t i = 0; i != nVis; ++i) {
    const double rms = static_cast<double>(metaOut[i % visPerRow]);
    if (rms != 0.0) {
      re[i] /= rms;
      im[i] /= rms;
    }
  }
  // K3 + K4 + K5b, polarization by polarization
  std::vector<double> matrix(nAnt * nAnt);
  std::vector<double> rms(nAnt);
  for (std::size_t pol = 0; pol != nPol; ++pol) {
    std::fill(matrix.begin(), matrix.end(), kNaN);
    for (std::size_t row = 0; row != nRows; ++row) {
      std::size_t a1 = static_cast<std::size_t>(ant1[row]);
      std::size_t a2 = static_cast<std::size_t>(ant2[row]);
      if (a1 == a2) continue;
      if (a1 > a2) std::swap(a1, a2);
      double value = 0.0;
      std::size_t count = 0;
      for (std::size_t ch = 0; ch != nChannels; ++ch) {
        const std::size_t idx = row * visPerRow + ch * nPol + pol;
        if (std::isfinite(re[idx]) && std::isfinite(im[idx])) {
          ++count;
          value += re[idx] * re[idx] + im[idx] * im[idx];
        }
      }
      const double r = std::sqrt(value / (count * 2.0));
      matrix[a1 * nAnt + a2] = r;
      matrix[a2 * nAnt + a1] = r;
    }
    solveAntennaGains(matrix, nAnt, rms);
    for (std::size_t a = 0; a != nAnt; ++a)
      metaOut[visPerRow + pol * nAnt + a] = static_cast<float>(rms[a]);
    for (std::size_t row = 0; row != nRows; ++row) {
      const double mul = rms[static_cast<std::size_t>(ant1[row])] *
                         rms[static_cast<std::size_t>(ant2[row])];
      const double fac = mul == 0.0 ? 0.0 : 1.0 / mul;
      for (std::size_t ch = 0; ch != nChannels; ++ch) {
        const std::size_t idx = row * visPerRow + ch * nPol + pol;
        re[idx] *= fac;
        im[idx] *= fac;
      }
    }
  }
  if (fitToMaximum)
    fitToMaximumRef(re, im, ant1, ant2, nChannels, nPol, nAnt, metaOut,
                    decCount == 0 ? 0.0
                                  : static_cast<double>(decValues[decCount - 1]));

  // K7
  const auto seedLo = static_cast<std::uint32_t>(seed);
  const auto seedHi = static_cast<std::uint32_t>(seed >> 32);
  for (std::size_t i = 0; i != nVis; ++i) {
    const auto base = static_cast<std::uint64_t>(i) * 2;
    symbolsOut[base] = quantizeDitherRef(
        static_cast<float>(re[i]), decValues,
        random123::ditherForVisibility(seedLo, seedHi, blockIndex, i, 0));
    symbolsOut[base + 1] = quantizeDitherRef(
        static_cast<float>(im[i]), decValues,
        random123::ditherForVisibility(seedLo, seedHi, blockIndex, i, 1));
  }
}

void cpuAfDecode(dysco::span<const std::uint32_t> symbols,
                 dysco::span<const float> meta,
                 dysco::span<const std::int32_t> ant1,
                 dysco::span<const std::int32_t> ant2, std::size_t nChannels,
                 std::size_t nPol, std::size_t nAnt,
                 dysco::span<const float> decValues,
                 dysco::span<float> interleavedOut) {
  const std::size_t nRows = ant1.size();
  const std::size_t visPerRow = nChannels * nPol;
  const std::size_t nVis = nRows * visPerRow;
  const std::size_t decCount = decValues.size();
  for (std::size_t i = 0; i != nVis; ++i) {
    const std::size_t row = i / visPerRow;
    const std::size_t rem = i % visPerRow;
    const std::size_t ch = rem / nPol;
    const std::size_t pol = rem % nPol;
    const double chRms = static_cast<double>(meta[ch * nPol + pol]);
    const double antFac =
        static_cast<double>(meta[visPerRow + pol * nAnt + ant1[row]]) *
        static_cast<double>(meta[visPerRow + pol * nAnt + ant2[row]]);
    const double factor = chRms * antFac;
    const std::uint32_t sRe = symbols[i * 2];
    const std::uint32_t sIm = symbols[i * 2 + 1];
    const double vRe =
        sRe < decCount ? static_cast<double>(decValues[sRe]) : kNaN;
    const double vIm =
        sIm < decCount ? static_cast<double>(decValues[sIm]) : kNaN;
    interleavedOut[i * 2] = static_cast<float>(vRe * factor);
    interleavedOut[i * 2 + 1] = static_cast<float>(vIm * factor);
  }
}

}  // namespace dyscostman::gpu
