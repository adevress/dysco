#include "aftimeblockencoder.h"

#include "kernels/af_reference.h"

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

AFTimeBlockEncoder::AFTimeBlockEncoder(size_t nPol, size_t nChannels,
                                       bool fitToMaximum)
    : _nPol(nPol),
      _nChannels(nChannels),
      _fitToMaximum(fitToMaximum),
      _rmsPerChannel(_nChannels * nPol) {}

AFTimeBlockEncoder::~AFTimeBlockEncoder() = default;

//
// There are 3 axes to maximize over; antenna1, antenna2, channel
// We want to maximize all values (average abs value as large as possible)
// Example: (10=max)
//  ch1     ch2
//  a1a2a3   a1a2a3
//  ------   ------
//   1 1 1    1 1 1
//  10 1 1    1 1 1
//   1 1 1    1 1 1
// ant3 * 10 -> sum = (3 * 1 + 10 + 5 * 10) + (4 * 1 + 5 * 10) = 117
//  ch2 * 10 -> sum = 18 + 90 = 108
// both * s10 -> sum = (13 + 5*s10) + (s10*4 + 10*5) = 91.5
//  ch1     ch2
//   1 1 1    1 1 1
//   1 1 1    1 1 1
//   1 2 1    1 1.5 1
// ant3 * 10 -> sum = (3 * 1 + 10 + 5 * 10) + (4 * 1 + 5 * 10) = 117
//  ch2 * 10 -> sum = 18 + 90 = 108
// both * s10 -> sum = (13 + 5*s10) + (s10*4 + 10*5) = 91.5
//
// Approach: iterate over all antenna and channels, and find the antenna/channel
// that can increase the sum the most.
//
// NOTE: the fit-to-maximum algorithm below used to live here.  It is now
// implemented once in kernels/af_reference.cc (fitToMaximum there), shared bit-
// exactly by the CPU AF encoder and the CUDA backend.
void AFTimeBlockEncoder::EncodeWithCounterRng(
    const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
    float *metaBuffer, symbol_t *symbolBuffer, size_t antennaCount,
    uint64_t seed, uint64_t blockIndex) {
  const auto &rows = buffer.GetVector();
  const std::size_t nRows = rows.size();
  const std::size_t visPerRow = _nPol * _nChannels;
  const std::size_t nVis = nRows * visPerRow;
  std::vector<float> interleaved(nVis * 2);
  std::vector<std::int32_t> ant1(nRows), ant2(nRows);
  for (std::size_t r = 0; r != nRows; ++r) {
    ant1[r] = static_cast<std::int32_t>(rows[r].antenna1);
    ant2[r] = static_cast<std::int32_t>(rows[r].antenna2);
    std::memcpy(interleaved.data() + r * visPerRow * 2,
                rows[r].visibilities.data(), visPerRow * 2 * sizeof(float));
  }
  const std::vector<float> decValues = gausEncoder.DecodeValues();
  const std::size_t metaCount = _nPol * (_nChannels + antennaCount);
  // Boundary cast: the storage manager symbol buffer is `symbol_t` (an
  // `unsigned`); the reference codec uses the fixed-width alias.
  static_assert(std::is_same_v<symbol_t, std::uint32_t>);
  dyscostman::gpu::cpuAfEncode(
      interleaved, ant1, ant2, _nChannels, _nPol, antennaCount, decValues, seed,
      blockIndex, _fitToMaximum, dysco::span<float>(metaBuffer, metaCount),
      dysco::span<std::uint32_t>(symbolBuffer, nVis * 2));
}

void AFTimeBlockEncoder::InitializeDecode(const float *metaBuffer,
                                          size_t /*nRow*/, size_t nAntennae) {
  if (_rmsPerAntenna.size() < nAntennae * _nPol)
    _rmsPerAntenna.resize(nAntennae * _nPol);
  const size_t visPerRow = _nPol * _nChannels;
  for (size_t i = 0; i != visPerRow; ++i) _rmsPerChannel[i] = metaBuffer[i];
  metaBuffer += visPerRow;
  for (size_t p = 0; p != _nPol; ++p) {
    for (size_t a = 0; a != nAntennae; ++a)
      _rmsPerAntenna[a * _nPol + p] = metaBuffer[a + p * nAntennae];
  }
}

void AFTimeBlockEncoder::Decode(
    const dyscostman::StochasticEncoder<float> &gausEncoder, FBuffer &buffer,
    const AFTimeBlockEncoder::symbol_t *symbolBuffer, size_t blockRow,
    size_t antenna1, size_t antenna2) {
  aocommon::UVector<double> antFactors(_nPol);
  for (size_t p = 0; p != _nPol; ++p)
    antFactors[p] = _rmsPerAntenna[antenna1 * _nPol + p] *
                    _rmsPerAntenna[antenna2 * _nPol + p];

  FBufferRow &row = buffer[blockRow];
  row.antenna1 = antenna1;
  row.antenna2 = antenna2;
  row.visibilities.resize(_nChannels * _nPol);
  std::complex<float> *destination = row.visibilities.data();
  const symbol_t *srcRowPtr = symbolBuffer + blockRow * SymbolsPerRow();
  for (size_t ch = 0; ch != _nChannels; ++ch) {
    for (size_t p = 0; p != _nPol; ++p) {
      double chRMS = _rmsPerChannel[ch * _nPol + p];
      double factor = chRMS * antFactors[p];
      destination->real(double(gausEncoder.Decode(*srcRowPtr)) * factor);
      ++srcRowPtr;
      destination->imag(double(gausEncoder.Decode(*srcRowPtr)) * factor);
      ++srcRowPtr;
      ++destination;
    }
  }
}
