// GPU variants of the unit tests that exercise code paths reimplemented by the
// CUDA backend (AF normalization / encode / decode, packing, dictionary
// quantization, dithering) plus storage-manager integration with the GPU
// backend enabled.
//
// The whole file is compiled only when the CUDA backend is enabled
// (DYSCO_HAS_CUDA, set by the ENABLE_CUDA CMake option).  Tests skip
// gracefully when no GPU device is present at runtime.

#include <boost/test/unit_test.hpp>

#ifdef DYSCO_HAS_CUDA

#include "../bytepacker.h"
#include "../dyscostman.h"
#include "../kernels/af_reference.h"
#include "../kernels/cuda/gpu_af_codec.h"
#include "../kernels/rng.h"
#include "../stochasticencoder.h"

// Boundary assertion: the production BytePacker (the storage format) is typed
// in terms of `unsigned int`, while the GPU codec uses the fixed-width alias.
static_assert(std::is_same_v<std::uint32_t, unsigned int>);

#include <casacore/casa/Arrays/IPosition.h>
#include <casacore/tables/DataMan/DataManager.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableDesc.h>

#include <boost/filesystem/directory.hpp>
#include <boost/filesystem/operations.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using namespace dyscostman;
using namespace dyscostman::gpu;

BOOST_AUTO_TEST_SUITE(gpu)

namespace {

bool HasGpu() { return available(); }

/// A synthetic AF time block in the SoA form the codec consumes.
struct GpuBlock {
  std::size_t nAnt = 0, nChan = 0, nPol = 0;
  std::size_t nRows = 0, visPerRow = 0, nVis = 0;
  std::vector<float> interleaved;  // 2*nVis
  std::vector<std::int32_t> ant1, ant2;
};

GpuBlock MakeLadderBlock(std::size_t nAnt, std::size_t nChan,
                         std::size_t nPol) {
  GpuBlock block;
  block.nAnt = nAnt;
  block.nChan = nChan;
  block.nPol = nPol;
  block.nRows = nAnt * (nAnt + 1) / 2;
  block.visPerRow = nChan * nPol;
  block.nVis = block.nRows * block.visPerRow;
  block.interleaved.resize(2 * block.nVis);
  std::size_t r = 0;
  for (std::size_t a = 0; a != nAnt; ++a)
    for (std::size_t b = a; b != nAnt; ++b) {
      block.ant1.push_back(static_cast<std::int32_t>(a));
      block.ant2.push_back(static_cast<std::int32_t>(b));
      ++r;
    }
  return block;
}

unsigned BitsFor(const StochasticEncoder<float>& encoder) {
  unsigned bits = 0;
  while ((1u << bits) < encoder.QuantizationCount()) ++bits;
  return bits;
}

/// Encode + decode on the GPU and return the interleaved decoded block.
std::vector<float> GpuRoundTrip(const GpuBlock& block,
                                const StochasticEncoder<float>& encoder,
                                std::uint64_t seed, bool fitToMaximum) {
  const unsigned bits = BitsFor(encoder);
  const std::vector<float> decValues = encoder.DecodeValues();
  AfCodec codec(block.nRows, block.nChan, block.nPol, block.nAnt, decValues,
                bits);
  std::vector<float> meta(block.visPerRow + block.nPol * block.nAnt);
  std::vector<std::uint8_t> packed(codec.packedByteCount());
  codec.encode(block.interleaved, block.ant1, block.ant2, seed, 0,
               fitToMaximum, meta, packed);
  std::vector<float> out(2 * block.nVis);
  codec.decode(packed, meta, block.ant1, block.ant2, out);
  return out;
}

/// Assert that the GPU encode/decode is bit-identical to the CPU reference.
void CheckGpuMatchesCpu(const GpuBlock& block,
                        const StochasticEncoder<float>& encoder,
                        std::uint64_t seed, bool fitToMaximum) {
  const unsigned bits = BitsFor(encoder);
  const std::vector<float> decValues = encoder.DecodeValues();
  const std::size_t metaCount = block.visPerRow + block.nPol * block.nAnt;
  const std::size_t symbolCount = 2 * block.nVis;

  std::vector<float> metaCpu(metaCount);
  std::vector<std::uint32_t> symbolsCpu(symbolCount);
  cpuAfEncode(block.interleaved, block.ant1, block.ant2, block.nChan, block.nPol,
              block.nAnt, decValues, seed, 0, fitToMaximum, metaCpu,
              symbolsCpu);
  std::vector<std::uint8_t> packedCpu((symbolCount * bits + 7) / 8);
  BytePacker::pack(bits, packedCpu.data(), symbolsCpu.data(), symbolCount);

  AfCodec codec(block.nRows, block.nChan, block.nPol, block.nAnt, decValues,
                bits);
  std::vector<float> metaGpu(metaCount);
  std::vector<std::uint8_t> packedGpu(codec.packedByteCount());
  codec.encode(block.interleaved, block.ant1, block.ant2, seed, 0,
               fitToMaximum, metaGpu, packedGpu);

  BOOST_CHECK_EQUAL(packedGpu.size(), packedCpu.size());

  if (!fitToMaximum) {
    // Without fit-to-max the GPU path is bit-identical to the CPU reference.
    BOOST_CHECK(std::memcmp(metaCpu.data(), metaGpu.data(),
                            metaCount * sizeof(float)) == 0);
    BOOST_CHECK(std::memcmp(packedCpu.data(), packedGpu.data(),
                            packedCpu.size()) == 0);

    // The GPU packed stream must unpack (with the production BytePacker) to
    // the CPU symbols, and the GPU decode must match the CPU decode.
    std::vector<std::uint32_t> symbolsFromGpu(symbolCount);
    BytePacker::unpack(bits, symbolsFromGpu.data(), packedGpu.data(),
                       symbolCount);
    BOOST_CHECK(symbolsFromGpu == symbolsCpu);

    std::vector<float> outCpu(2 * block.nVis), outGpu(2 * block.nVis);
    cpuAfDecode(symbolsCpu, metaCpu, block.ant1, block.ant2, block.nChan,
                block.nPol, block.nAnt, decValues, outCpu);
    codec.decode(packedGpu, metaGpu, block.ant1, block.ant2, outGpu);
    BOOST_CHECK(outGpu == outCpu);
    return;
  }

  // Fit-to-max uses the parallel Jacobi algorithm on the GPU while the CPU
  // encoder deliberately keeps the greedy algorithm, so the two are not
  // bit-identical (different headroom allocation).  Validate that they agree
  // within a few percent instead.
  for (std::size_t i = 0; i != metaCount; ++i) {
    const float a = metaCpu[i], b = metaGpu[i];
    const float tolerance = 1e-2f * std::max(1.0f, std::fabs(a));
    BOOST_CHECK_MESSAGE(std::fabs(a - b) <= tolerance,
                        "metadata[" << i << "]: " << a << " vs " << b);
  }
  std::vector<float> outCpu(2 * block.nVis), outGpu(2 * block.nVis);
  cpuAfDecode(symbolsCpu, metaCpu, block.ant1, block.ant2, block.nChan,
              block.nPol, block.nAnt, decValues, outCpu);
  codec.decode(packedGpu, metaGpu, block.ant1, block.ant2, outGpu);
  double error = 0.0, signal = 0.0;
  for (std::size_t i = 0; i != outCpu.size(); ++i) {
    const double d = static_cast<double>(outGpu[i]) - outCpu[i];
    error += d * d;
    signal += static_cast<double>(outCpu[i]) * outCpu[i];
  }
  BOOST_CHECK_LT(std::sqrt(error), std::sqrt(signal) * 1e-2 + 1e-4);
}

float Rms(const std::vector<float>& values) {
  double sum = 0.0;
  for (const float v : values) sum += static_cast<double>(v) * v;
  return static_cast<float>(std::sqrt(sum / values.size()));
}

}  // namespace

// ---------------------------------------------------------------------------
// AF normalization (mirrors timeblock_encoder/af_*)
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(af_normalization_per_row_accuracy) {
  if (!HasGpu()) return;
  const std::size_t nAnt = 4, nChan = 1, nPol = 2;
  GpuBlock block = MakeLadderBlock(nAnt, nChan, nPol);
  const std::size_t visPerRow = block.visPerRow;
  // Same fixture data as the CPU TestSimpleExample.
  const std::complex<float> data[10][2] = {
      {99.0f, 99.0f},
      {10.0f, {9.0f, 1.0f}},
      {8.0f, {7.0f, 2.0f}},
      {6.0f, {5.0f, 3.0f}},
      {99.0f, 99.0f},
      {4.0f, {3.0f, 4.0f}},
      {2.0f, {1.0f, 5.0f}},
      {99.0f, 99.0f},
      {0.0f, std::complex<float>(std::numeric_limits<float>::quiet_NaN(),
                                 std::numeric_limits<float>::quiet_NaN())},
      {99.0f, 99.0f}};
  for (std::size_t row = 0; row != block.nRows; ++row)
    for (std::size_t i = 0; i != visPerRow; ++i) {
      block.interleaved[(row * visPerRow + i) * 2] = data[row][i].real();
      block.interleaved[(row * visPerRow + i) * 2 + 1] = data[row][i].imag();
    }

  StochasticEncoder<float> encoder(256, 1.0, false);
  const std::vector<float> out = GpuRoundTrip(block, encoder, 1234, true);

  for (std::size_t row = 0; row != block.nRows; ++row) {
    // skip auto-correlations of AF, since these are not saved
    if (row == 0 || row == 4 || row == 7 || row == 9) continue;
    for (std::size_t ch = 0; ch != nChan; ++ch) {
      const std::complex<float> decoded(out[(row * visPerRow + ch) * 2],
                                        out[(row * visPerRow + ch) * 2 + 1]);
      const std::complex<float> expected = data[row][ch];
      BOOST_CHECK_MESSAGE(std::norm(decoded - expected) < 0.1,
                          "GPU AF row " << row << " decoded " << decoded
                                        << " vs input " << expected);
    }
  }
}

BOOST_AUTO_TEST_CASE(af_normalization_global_rms_accuracy) {
  if (!HasGpu()) return;
  const std::size_t nAnt = 50, nChan = 64, nPol = 4;
  GpuBlock block = MakeLadderBlock(nAnt, nChan, nPol);

  std::mt19937 rnd(42);
  std::normal_distribution<float> dist;
  for (std::size_t i = 0; i != block.nVis; ++i) {
    block.interleaved[i * 2] = dist(rnd);
    block.interleaved[i * 2 + 1] = dist(rnd);
  }

  StochasticEncoder<float> encoder(256, 1.0, false);
  const std::vector<float> out = GpuRoundTrip(block, encoder, 7, true);

  std::vector<float> error(block.nVis), signal(block.nVis);
  for (std::size_t i = 0; i != block.nVis; ++i) {
    error[i] = std::hypot(out[i * 2] - block.interleaved[i * 2],
                          out[i * 2 + 1] - block.interleaved[i * 2 + 1]);
    signal[i] = std::hypot(block.interleaved[i * 2],
                           block.interleaved[i * 2 + 1]);
  }
  BOOST_CHECK_LT(Rms(error), Rms(signal) * 0.1f);

  CheckGpuMatchesCpu(block, encoder, 7, true);
}

BOOST_AUTO_TEST_CASE(af_normalization_with_zeros) {
  if (!HasGpu()) return;
  const std::size_t nAnt = 4, nChan = 1, nPol = 2;
  GpuBlock block = MakeLadderBlock(nAnt, nChan, nPol);
  block.interleaved.assign(2 * block.nVis, 0.0f);

  StochasticEncoder<float> encoder(256, 1.0, false);
  const std::vector<float> out = GpuRoundTrip(block, encoder, 3, true);

  for (std::size_t row = 0; row != block.nRows; ++row) {
    if (row == 0 || row == 4 || row == 7 || row == 9) continue;
    for (std::size_t i = 0; i != block.visPerRow; ++i) {
      BOOST_CHECK(std::isfinite(out[(row * block.visPerRow + i) * 2]));
      BOOST_CHECK(std::isfinite(out[(row * block.visPerRow + i) * 2 + 1]));
      BOOST_CHECK_EQUAL(out[(row * block.visPerRow + i) * 2], 0.0f);
      BOOST_CHECK_EQUAL(out[(row * block.visPerRow + i) * 2 + 1], 0.0f);
    }
  }
  CheckGpuMatchesCpu(block, encoder, 3, true);
}

// ---------------------------------------------------------------------------
// bytepacking (K8 / D1) — mirrors bytepacking/*
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(bytepacking) {
  if (!HasGpu()) return;
  const unsigned bitWidths[] = {2, 3, 4, 6, 8, 10, 12, 16};
  std::mt19937 rnd(1);
  for (const unsigned bits : bitWidths) {
    for (std::size_t count : {1u, 7u, 15u, 31u, 64u, 257u, 1000u}) {
      std::vector<std::uint32_t> symbols(count);
      const unsigned limit = (1u << bits);
      for (std::size_t i = 0; i != count; ++i)
        symbols[i] = rnd() % limit;

      const std::size_t packedBytes = (count * bits + 7) / 8;
      std::vector<std::uint8_t> packedGpu(packedBytes, 0xAA);
      pack(symbols, bits, packedGpu);

      std::vector<std::uint8_t> packedCpu(packedBytes, 0xAA);
      BytePacker::pack(bits, packedCpu.data(), symbols.data(), count);
      BOOST_CHECK_MESSAGE(packedGpu == packedCpu,
                          "bits=" << bits << " count=" << count);

      std::vector<std::uint32_t> restoredGpu(count, 0);
      unpack(packedGpu, bits, restoredGpu);
      BOOST_CHECK_MESSAGE(restoredGpu == symbols,
                          "bits=" << bits << " count=" << count);
    }
  }
}

// ---------------------------------------------------------------------------
// dictionary / lower_bound (K7) — mirrors dictionary/lower_bound*
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(dictionary_lower_bound) {
  if (!HasGpu()) return;
  std::mt19937 rnd(5);
  std::normal_distribution<float> dist;

  for (unsigned bits : {4u, 8u, 12u}) {
    StochasticEncoder<float> encoder =
        StochasticEncoder<float>::TruncatedGausEncoder(1u << bits, 2.5, 1.0);
    const std::vector<float> decValues = encoder.DecodeValues();

    std::vector<float> values;
    for (float v : {-10.0f, -1.0f, -0.001f, 0.0f, 0.001f, 1.0f, 10.0f})
      values.push_back(v);
    for (int i = 0; i != 2000; ++i) values.push_back(dist(rnd));
    for (int i = 0; i != 50; ++i)
      values.push_back(std::numeric_limits<float>::quiet_NaN());
    values.push_back(std::numeric_limits<float>::infinity());

    const std::uint64_t seed = 0x0123456789ABCDEFull;
    std::vector<std::uint32_t> expected(values.size());
    for (std::size_t i = 0; i != values.size(); ++i) {
      const auto seedLo = static_cast<std::uint32_t>(seed);
      const auto seedHi = static_cast<std::uint32_t>(seed >> 32);
      const std::uint32_t dither =
          rng::ditherValue(seedLo, seedHi, 0, i);
      expected[i] = encoder.EncodeWithDithering(values[i], dither);
    }

    std::vector<std::uint32_t> gpu(values.size());
    quantize(values, decValues, seed, 0, gpu);
    BOOST_CHECK_MESSAGE(gpu == expected, "dictionary mismatch at bits=" << bits);
  }
}

// ---------------------------------------------------------------------------
// dithering statistics (K7 + CBRNG) — mirrors dithering/avg_deviation
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(dithering_avg_deviation) {
  if (!HasGpu()) return;
  StochasticEncoder<float> encoder =
      StochasticEncoder<float>::StudentTEncoder(256, 1.0, 1.0);
  const std::vector<float> decValues = encoder.DecodeValues();

  const std::vector<float> probes = {-static_cast<float>(M_PI),
                                     -static_cast<float>(M_E),
                                     -static_cast<float>(M_SQRT2),
                                     -1.0f,
                                     -0.01f,
                                     0.0f,
                                     0.01f,
                                     1.0f,
                                     static_cast<float>(M_SQRT2),
                                     static_cast<float>(M_E),
                                     static_cast<float>(M_PI)};
  const std::size_t repetitions = 100000;
  const std::size_t n = probes.size() * repetitions;

  std::vector<float> values(n);
  for (std::size_t r = 0; r != repetitions; ++r)
    std::copy(probes.begin(), probes.end(), values.begin() + r * probes.size());

  std::vector<std::uint32_t> symbols(n);
  quantize(values, decValues, 424242, 0, symbols);

  std::vector<long double> sums(probes.size(), 0.0L);
  for (std::size_t r = 0; r != repetitions; ++r)
    for (std::size_t t = 0; t != probes.size(); ++t)
      sums[t] += encoder.Decode(symbols[r * probes.size() + t]);

  for (std::size_t t = 0; t != probes.size(); ++t) {
    const double mean = static_cast<double>(sums[t] / repetitions);
    BOOST_CHECK_LE(std::fabs(mean - probes[t]), 0.05);
  }
}

// ---------------------------------------------------------------------------
// encode_example (StochasticEncoder + dither) — mirrors encode_example/*,
// without writing files to disk.
// ---------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(encode_example) {
  if (!HasGpu()) return;
  StochasticEncoder<float> encoder =
      StochasticEncoder<float>::TruncatedGausEncoder(16, 2.0, 1.0);
  const std::vector<float> decValues = encoder.DecodeValues();

  auto check = [&](const std::vector<float>& probes) {
    const std::size_t repetitions = 200;
    std::vector<float> values;
    values.reserve(probes.size() * repetitions);
    for (std::size_t r = 0; r != repetitions; ++r)
      values.insert(values.end(), probes.begin(), probes.end());
    std::vector<std::uint32_t> symbols(values.size());
    quantize(values, decValues, 999, 0, symbols);
    std::vector<long double> sums(probes.size(), 0.0L);
    for (std::size_t r = 0; r != repetitions; ++r)
      for (std::size_t t = 0; t != probes.size(); ++t)
        sums[t] += encoder.Decode(symbols[r * probes.size() + t]);
    for (std::size_t t = 0; t != probes.size(); ++t) {
      if (std::fabs(probes[t]) < encoder.MaxQuantity()) {
        const double mean = static_cast<double>(sums[t] / repetitions);
        BOOST_CHECK_LT(std::fabs(mean - probes[t]), 0.1);
      }
    }
  };

  constexpr std::size_t n = 100;
  std::vector<float> sinus(n), oneOver(n), sinc(n);
  for (std::size_t i = 0; i != n; ++i) {
    const double x = static_cast<double>(i) * 2 * M_PI / n;
    sinus[i] = static_cast<float>(std::sin(x));
    oneOver[i] = static_cast<float>(1.0 / (12.0 * static_cast<double>(i) / n));
    const double sx = static_cast<double>(i) / n * 2.5 * M_PI;
    sinc[i] = i == 0 ? 1.0f : static_cast<float>(std::sin(sx) / sx);
  }
  check(sinus);
  check(oneOver);
  check(sinc);
}

// ---------------------------------------------------------------------------
// storage manager integration — mirrors dyscostman/{spec,name,makecolumn,
// maketable,readonly} with the GPU backend enabled.
// ---------------------------------------------------------------------------

namespace {

casacore::Record GpuDyscoSpec() {
  casacore::Record spec;
  spec.define("distribution", "TruncatedGaussian");
  spec.define("normalization", "AF");
  spec.define("distributionTruncation", 2.0);
  spec.define("dataBitCount", 10);
  spec.define("weightBitCount", 12);
  return spec;
}

/// Creates `TestTableGpu` with a GPU-enabled DyscoStMan DATA column.
struct GpuTableFixture {
  explicit GpuTableFixture(std::size_t nAnt) {
    casacore::TableDesc tableDesc;
    const casacore::IPosition shape(2, 1, 1);
    casacore::ArrayColumnDesc<casacore::Complex> columnDesc("DATA", "",
                                                            "DyscoStMan", "",
                                                            shape);
    columnDesc.setOptions(casacore::ColumnDesc::Direct |
                          casacore::ColumnDesc::FixedShape);
    casacore::ScalarColumnDesc<int> ant1Desc("ANTENNA1"), ant2Desc("ANTENNA2"),
        fieldDesc("FIELD_ID"), dataDescIdDesc("DATA_DESC_ID");
    casacore::ScalarColumnDesc<double> timeDesc("TIME");
    tableDesc.addColumn(columnDesc);
    tableDesc.addColumn(ant1Desc);
    tableDesc.addColumn(ant2Desc);
    tableDesc.addColumn(fieldDesc);
    tableDesc.addColumn(dataDescIdDesc);
    tableDesc.addColumn(timeDesc);
    casacore::SetupNewTable setupNewTable("TestTableGpu", tableDesc,
                                          casacore::Table::New);

    register_dyscostman();
    // The table takes ownership of the manager (as in testdyscostman.cc).
    std::unique_ptr<DyscoStMan> manager(
        new DyscoStMan("DATA_dm", GpuDyscoSpec()));
    manager->SetGpuEnabled(true);
    setupNewTable.bindColumn("DATA", *manager);
    casacore::Table newTable(setupNewTable);

    const std::size_t nRow = 2 * nAnt * (nAnt - 1) / 2;
    newTable.addRow(nRow);
    casacore::ScalarColumn<int> a1Col(newTable, "ANTENNA1"),
        a2Col(newTable, "ANTENNA2"), fieldCol(newTable, "FIELD_ID"),
        dataDescIdCol(newTable, "DATA_DESC_ID");
    casacore::ScalarColumn<double> timeCol(newTable, "TIME");
    std::size_t a1 = 0, a2 = 1;
    double time = 10.0;
    for (std::size_t i = 0; i != nRow; ++i) {
      a1Col.put(i, static_cast<int>(a1));
      a2Col.put(i, static_cast<int>(a2));
      fieldCol.put(i, 0);
      dataDescIdCol.put(i, 0);
      timeCol.put(i, time);
      ++a2;
      if (a2 == nAnt) {
        ++a1;
        a2 = a1 + 1;
        if (a2 == nAnt) {
          a1 = 0;
          a2 = 1;
          ++time;
        }
      }
    }

    casacore::ArrayColumn<casacore::Complex> dataCol(newTable, "DATA");
    for (std::size_t i = 0; i != nRow; ++i) {
      casacore::Array<casacore::Complex> arr(shape);
      *arr.cbegin() = static_cast<float>(i);
      dataCol.put(i, arr);
    }
  }

  ~GpuTableFixture() { boost::filesystem::remove_all("TestTableGpu"); }
};

}  // namespace

BOOST_AUTO_TEST_CASE(dyscostman_spec_name) {
  if (!HasGpu()) return;
  DyscoStMan dysco(8, 12, "gpu_manager");
  dysco.SetGpuEnabled(true);
  const casacore::Record spec = dysco.dataManagerSpec();
  BOOST_CHECK_EQUAL(spec.asInt("dataBitCount"), 8);
  BOOST_CHECK_EQUAL(spec.asInt("weightBitCount"), 12);
  BOOST_CHECK_EQUAL(dysco.dataManagerType(), "DyscoStMan");
  BOOST_CHECK_EQUAL(dysco.dataManagerName(), "gpu_manager");
  std::unique_ptr<casacore::DataManager> clone(dysco.clone());
  BOOST_CHECK_EQUAL(clone->dataManagerName(), "gpu_manager");
}

BOOST_AUTO_TEST_CASE(dyscostman_makecolumn) {
  if (!HasGpu()) return;
  DyscoStMan dysco(8, 12, "gpu_manager");
  dysco.SetGpuEnabled(true);
  dysco.createDirArrColumn("DATA", casacore::DataType::TpComplex, "");
  dysco.createDirArrColumn("ANYTHING", casacore::DataType::TpComplex, "");
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(dyscostman_maketable) {
  if (!HasGpu()) return;
  GpuTableFixture fixture(3);
  casacore::Table table("TestTableGpu");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  // The GPU fit path uses the parallel Jacobi algorithm, a heuristic that is
  // not bit-identical to the CPU greedy metadadata (see
  // instructions-fit-to-max-gpu.md); allow a 1 % tolerance.
  for (std::size_t i = 0; i != table.nrow(); ++i)
    BOOST_CHECK_CLOSE_FRACTION((*dataCol(i).cbegin()).real(),
                               static_cast<float>(i), 1e-2);
}

BOOST_AUTO_TEST_CASE(dyscostman_readonly) {
  if (!HasGpu()) return;
  GpuTableFixture fixture(3);
  for (boost::filesystem::directory_iterator itr("TestTableGpu/"),
       end;
       itr != end; ++itr) {
    if (boost::filesystem::is_regular_file(itr->path()))
      boost::filesystem::permissions(
          itr->path(),
          boost::filesystem::others_read | boost::filesystem::owner_read);
  }
  casacore::Table table("TestTableGpu");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  for (std::size_t i = 0; i != table.nrow(); ++i)
    BOOST_CHECK_CLOSE_FRACTION((*dataCol(i).cbegin()).real(),
                               static_cast<float>(i), 1e-2);
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // DYSCO_HAS_CUDA
