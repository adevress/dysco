#include "af_reference.h"

#include "cuda/philox_rng.h"

#include <cmath>
#include <limits>
#include <vector>

namespace dyscostman {
namespace gpu {

namespace {

const double kNaN = std::numeric_limits<double>::quiet_NaN();

/// K7 helper: the CPU mirror of quantizeDither().
unsigned int quantizeDitherRef(float value, const float *decValues,
                               size_t decCount, unsigned int dither) {
  if (!std::isfinite(value))
    return static_cast<unsigned int>(decCount);
  size_t lo = 0, hi = decCount;
  while (lo < hi) {
    const size_t mid = (lo + hi) >> 1;
    if (decValues[mid] < value)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0) return 0;
  if (lo >= decCount) return static_cast<unsigned int>(decCount - 1);
  const float rightValue = decValues[lo];
  const float leftValue = decValues[lo - 1];
  const float ditherMark =
      static_cast<float>(2147483648.0f * (value - leftValue) /
                         (rightValue - leftValue));
  if (ditherMark > static_cast<float>(dither))
    return static_cast<unsigned int>(lo);
  return static_cast<unsigned int>(lo - 1);
}

/// K4 mirror.
void solveAntennaGains(const std::vector<double> &matrix, size_t nAnt,
                       std::vector<double> &rms) {
  for (size_t i = 0; i != nAnt; ++i) rms[i] = 1.0;
  double precision = 1.0;
  for (size_t iteration = 0; iteration != 100 && precision > 1e-6;
       ++iteration) {
    std::vector<double> next(nAnt, 0.0);
    for (size_t i = 0; i != nAnt; ++i) {
      double weightSum = 0.0, acc = 0.0;
      for (size_t j = 0; j != nAnt; ++j) {
        if (i != j && rms[j] != 0.0) {
          const double w = rms[j];
          const double m = matrix[i * nAnt + j];
          if (std::isfinite(m)) {
            acc += m;
            weightSum += w;
          }
        }
      }
      next[i] = (weightSum == 0.0) ? 0.0 : acc / weightSum;
    }
    double maxVal = 0.0;
    for (size_t i = 0; i != nAnt; ++i) {
      rms[i] = next[i] * 0.8 + rms[i] * 0.2;
      maxVal = std::max(maxVal, rms[i]);
    }
    precision = 0.0;
    for (size_t i = 0; i != nAnt; ++i) {
      if (rms[i] < maxVal * 1e-5) rms[i] = 0.0;
      precision = std::max(precision, std::fabs(rms[i] - next[i]) / maxVal);
    }
  }
}

/// K6 mirror.
void fitToMaximumRef(std::vector<double> &re, std::vector<double> &im,
                     const std::vector<int> &ant1, const std::vector<int> &ant2,
                     size_t nChannels, size_t nPol, size_t nAnt, float *meta,
                     double maxLevel) {
  const size_t visPerRow = nChannels * nPol;
  const size_t nRows = ant1.size();

  auto changeChannelFactor = [&](size_t visIndex, double factor) {
    meta[visIndex] =
        static_cast<float>(static_cast<double>(meta[visIndex]) / factor);
    for (size_t row = 0; row != nRows; ++row) {
      const size_t idx = row * visPerRow + visIndex;
      re[idx] *= factor;
      im[idx] *= factor;
    }
  };
  auto changeAntennaFactor = [&](size_t antennaIndex, size_t polIndex,
                                 double factor) {
    meta[visPerRow + nAnt * polIndex + antennaIndex] = static_cast<float>(
        static_cast<double>(meta[visPerRow + nAnt * polIndex + antennaIndex]) /
        factor);
    for (size_t row = 0; row != nRows; ++row) {
      const size_t a1 = static_cast<size_t>(ant1[row]);
      const size_t a2 = static_cast<size_t>(ant2[row]);
      unsigned count = 0;
      if (a1 == antennaIndex) ++count;
      if (a2 == antennaIndex) ++count;
      for (unsigned repeat = 0; repeat != count; ++repeat)
        for (size_t ch = 0; ch != nChannels; ++ch) {
          const size_t idx = row * visPerRow + ch * nPol + polIndex;
          re[idx] *= factor;
          im[idx] *= factor;
        }
    }
  };

  for (size_t visIndex = 0; visIndex != visPerRow; ++visIndex) {
    double largest_component = 0.0;
    for (size_t row = 0; row != nRows; ++row) {
      if (ant1[row] != ant2[row]) {
        const size_t idx = row * visPerRow + visIndex;
        const double local_max =
            std::max(std::max(re[idx], im[idx]), -std::min(re[idx], im[idx]));
        if (std::isfinite(local_max) && local_max > largest_component)
          largest_component = local_max;
      }
    }
    const double factor =
        (maxLevel == 0.0 || largest_component == 0.0)
            ? 1.0
            : maxLevel / largest_component;
    changeChannelFactor(visIndex, factor);
  }

  for (size_t polIndex = 0; polIndex != nPol; ++polIndex) {
    bool isProgressing;
    do {
      double bestChannelIncrease = 0.0, channelFactor = 1.0;
      size_t bestChannel = 0;
      std::vector<double> largestCh(nChannels, 0.0);
      for (size_t channel = 0; channel != nChannels; ++channel) {
        const size_t visIndex = channel * nPol + polIndex;
        double largest_component = 0.0;
        for (size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const size_t idx = row * visPerRow + visIndex;
            const double local_max = std::max(std::max(re[idx], im[idx]),
                                              -std::min(re[idx], im[idx]));
            if (std::isfinite(local_max) && local_max > largest_component)
              largest_component = local_max;
          }
        }
        largestCh[channel] = largest_component;
        const double factor =
            (largest_component == 0.0) ? 0.0 : (maxLevel / largest_component - 1.0);
        double thisIncrease = 0.0;
        for (size_t row = 0; row != nRows; ++row) {
          if (ant1[row] != ant2[row]) {
            const size_t idx = row * visPerRow + visIndex;
            const double av = std::fabs(re[idx] * factor) +
                              std::fabs(im[idx] * factor);
            if (std::isfinite(av)) thisIncrease += av;
          }
        }
        if (thisIncrease > bestChannelIncrease) {
          bestChannelIncrease = thisIncrease;
          bestChannel = channel;
          channelFactor = factor + 1.0;
        }
      }

      std::vector<double> maxCompPerAntenna(nAnt, 0.0);
      for (size_t row = 0; row != nRows; ++row) {
        if (ant1[row] == ant2[row]) continue;
        const size_t a1 = static_cast<size_t>(ant1[row]);
        const size_t a2 = static_cast<size_t>(ant2[row]);
        for (size_t channel = 0; channel != nChannels; ++channel) {
          const size_t idx = row * visPerRow + channel * nPol + polIndex;
          const double local_max =
              std::max(std::max(re[idx], im[idx]), -std::min(re[idx], im[idx]));
          if (std::isfinite(local_max)) {
            if (local_max > maxCompPerAntenna[a1])
              maxCompPerAntenna[a1] = local_max;
            if (local_max > maxCompPerAntenna[a2])
              maxCompPerAntenna[a2] = local_max;
          }
        }
      }
      std::vector<double> increasePerAntenna(nAnt, 0.0);
      for (size_t row = 0; row != nRows; ++row) {
        if (ant1[row] == ant2[row]) continue;
        const size_t a1 = static_cast<size_t>(ant1[row]);
        const size_t a2 = static_cast<size_t>(ant2[row]);
        const double factor1 =
            (maxCompPerAntenna[a1] == 0.0)
                ? 0.0
                : (maxLevel / maxCompPerAntenna[a1] - 1.0);
        const double factor2 =
            (maxCompPerAntenna[a2] == 0.0)
                ? 0.0
                : (maxLevel / maxCompPerAntenna[a2] - 1.0);
        for (size_t channel = 0; channel != nChannels; ++channel) {
          const size_t idx = row * visPerRow + channel * nPol + polIndex;
          const double av1 =
              std::fabs(re[idx] * factor1) + std::fabs(im[idx] * factor1);
          if (std::isfinite(av1)) increasePerAntenna[a1] += av1;
          const double av2 =
              std::fabs(re[idx] * factor2) + std::fabs(im[idx] * factor2);
          if (std::isfinite(av2)) increasePerAntenna[a2] += av2;
        }
      }
      size_t bestAntenna = 0;
      double bestAntennaIncrease = 0.0;
      for (size_t a = 0; a != nAnt; ++a) {
        if (increasePerAntenna[a] > bestAntennaIncrease) {
          bestAntennaIncrease = increasePerAntenna[a];
          bestAntenna = a;
        }
      }
      if (bestAntennaIncrease > bestChannelIncrease) {
        const double factor =
            (maxCompPerAntenna[bestAntenna] == 0.0)
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

void cpuAfEncode(const float *interleaved, const int *ant1, const int *ant2,
                 size_t nRows, size_t nChannels, size_t nPol, size_t nAnt,
                 const float *decValues, size_t decCount, uint64_t seed,
                 uint64_t blockIndex, bool fitToMaximum, float *metaOut,
                 unsigned int *symbolsOut) {
  const size_t visPerRow = nChannels * nPol;
  const size_t nVis = nRows * visPerRow;
  std::vector<double> re(nVis), im(nVis);
  std::vector<int> hAnt1(ant1, ant1 + nRows), hAnt2(ant2, ant2 + nRows);

  // K1
  for (size_t i = 0; i != nVis; ++i) {
    re[i] = static_cast<double>(interleaved[i * 2]);
    im[i] = static_cast<double>(interleaved[i * 2 + 1]);
  }
  // K2
  for (size_t i = 0; i != visPerRow; ++i) {
    double value = 0.0;
    size_t count = 0;
    for (size_t row = 0; row != nRows; ++row) {
      const size_t idx = row * visPerRow + i;
      if (std::isfinite(re[idx]) && std::isfinite(im[idx])) {
        ++count;
        value += re[idx] * re[idx] + im[idx] * im[idx];
      }
    }
    metaOut[i] = static_cast<float>(std::sqrt(value / (count * 2.0)));
  }
  // K5a
  for (size_t i = 0; i != nVis; ++i) {
    const double rms = static_cast<double>(metaOut[i % visPerRow]);
    re[i] /= rms;
    im[i] /= rms;
  }
  // K3 + K4 + K5b, polarity by polarity
  std::vector<double> matrix(nAnt * nAnt, kNaN);
  std::vector<double> rms(nAnt);
  for (size_t pol = 0; pol != nPol; ++pol) {
    std::fill(matrix.begin(), matrix.end(), kNaN);
    for (size_t row = 0; row != nRows; ++row) {
      size_t a1 = static_cast<size_t>(hAnt1[row]);
      size_t a2 = static_cast<size_t>(hAnt2[row]);
      if (a1 == a2) continue;
      if (a1 > a2) std::swap(a1, a2);
      double value = 0.0;
      size_t count = 0;
      for (size_t ch = 0; ch != nChannels; ++ch) {
        const size_t idx = row * visPerRow + ch * nPol + pol;
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
    for (size_t a = 0; a != nAnt; ++a)
      metaOut[visPerRow + pol * nAnt + a] = static_cast<float>(rms[a]);
    for (size_t row = 0; row != nRows; ++row) {
      const double mul = rms[static_cast<size_t>(hAnt1[row])] *
                         rms[static_cast<size_t>(hAnt2[row])];
      const double fac = (mul == 0.0) ? 0.0 : 1.0 / mul;
      for (size_t ch = 0; ch != nChannels; ++ch) {
        const size_t idx = row * visPerRow + ch * nPol + pol;
        re[idx] *= fac;
        im[idx] *= fac;
      }
    }
  }
  if (fitToMaximum)
    fitToMaximumRef(re, im, hAnt1, hAnt2, nChannels, nPol, nAnt, metaOut,
                    decCount == 0 ? 0.0
                                  : static_cast<double>(decValues[decCount - 1]));

  // K7
  const uint32_t seedLo = static_cast<uint32_t>(seed);
  const uint32_t seedHi = static_cast<uint32_t>(seed >> 32);
  for (size_t i = 0; i != nVis; ++i) {
    const uint64_t base = static_cast<uint64_t>(i) * 2;
    const unsigned int dRe =
        random123::ditherValue(seedLo, seedHi, blockIndex, base);
    const unsigned int dIm =
        random123::ditherValue(seedLo, seedHi, blockIndex, base + 1);
    symbolsOut[base] = quantizeDitherRef(static_cast<float>(re[i]), decValues,
                                         decCount, dRe);
    symbolsOut[base + 1] = quantizeDitherRef(static_cast<float>(im[i]),
                                             decValues, decCount, dIm);
  }
}

void cpuAfDecode(const unsigned int *symbols, const float *meta,
                 const int *ant1, const int *ant2, size_t nRows,
                 size_t nChannels, size_t nPol, size_t nAnt,
                 const float *decValues, size_t decCount,
                 float *interleavedOut) {
  const size_t visPerRow = nChannels * nPol;
  const size_t nVis = nRows * visPerRow;
  for (size_t i = 0; i != nVis; ++i) {
    const size_t row = i / visPerRow;
    const size_t rem = i % visPerRow;
    const size_t ch = rem / nPol;
    const size_t pol = rem % nPol;
    const double chRms = static_cast<double>(meta[ch * nPol + pol]);
    const double antFac =
        static_cast<double>(meta[visPerRow + pol * nAnt + ant1[row]]) *
        static_cast<double>(meta[visPerRow + pol * nAnt + ant2[row]]);
    const double factor = chRms * antFac;
    const unsigned int sRe = symbols[i * 2];
    const unsigned int sIm = symbols[i * 2 + 1];
    const double vRe =
        (sRe < decCount) ? static_cast<double>(decValues[sRe]) : kNaN;
    const double vIm =
        (sIm < decCount) ? static_cast<double>(decValues[sIm]) : kNaN;
    interleavedOut[i * 2] = static_cast<float>(vRe * factor);
    interleavedOut[i * 2 + 1] = static_cast<float>(vIm * factor);
  }
}

}  // namespace gpu
}  // namespace dyscostman
