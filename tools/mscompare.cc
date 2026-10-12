// mscompare -- detailed error statistics between an original measurement set
// and a (decompressed) copy of it.
//
// Usage:
//   mscompare [options] <reference.ms> <test.ms>
//
// Options:
//   -column <name>   column to compare (default: DATA; repeatable)
//   -per-channel     also print per-(channel,pol) RMS error
//   -csv             print one CSV summary line per column instead of tables
//   -q               quiet: only the per-column header
//
// Only *unflagged* samples are compared.  Flagged samples are excluded because
// the compressor replaces them with NaN before encoding and the flag column is
// the authoritative record of which samples are meaningful.  Any unflagged
// sample that is non-finite in either input is counted as a validity mismatch
// and excluded from the moment statistics.
//
// Reported quantities (over one column, complex samples x = (re, im)):
//   n            number of compared samples (unflagged, both finite)
//   flagged      number of flagged samples
//   refNaN/testNaN  unflagged samples that are non-finite in ref / test
//   maxAbsErr    max |test - ref|
//   maxErrRe/Im  max |reTest-reRef| / |imTest-imRef|
//   meanErrRe/Im mean signed error per component (bias)
//   stdErrRe/Im  standard deviation of the signed error per component
//   mseRe/mseIm  mean squared error per component
//   mse          mean |test-ref|^2
//   rmse         sqrt(mse)
//   mae          mean |test-ref|
//   sigRms       RMS amplitude of the reference samples
//   relRmse      rmse / sigRms
//   snrDb        10 log10(signal power / noise power)
//   maxRef       max |ref| (component-wise)
//   p50/p90/p99/p999  percentiles of |test-ref|

#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/ScalarColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableDesc.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::vector<std::string> columns{"DATA"};
  bool perChannel = false;
  bool perRow = false;
  long worst = 0;
  bool csv = false;
  bool quiet = false;
  std::string ref;
  std::string test;
};

[[noreturn]] void usage(const char *arg0) {
  std::fprintf(stderr,
               "Usage: %s [-column <name>]... [-per-channel] [-per-row] "
               "[-worst N] [-csv] [-q] <reference.ms> <test.ms>\n",
               arg0);
  std::exit(2);
}

Options parse(int argc, char **argv) {
  Options o;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    const std::string a(argv[i]);
    if (a == "-column") {
      if (i + 1 >= argc) usage(argv[0]);
      if (o.columns.size() == 1 && o.columns[0] == "DATA") o.columns.clear();
      o.columns.push_back(argv[++i]);
    } else if (a == "-per-channel") {
      o.perChannel = true;
    } else if (a == "-per-row") {
      o.perRow = true;
    } else if (a == "-worst") {
      if (i + 1 >= argc) usage(argv[0]);
      o.worst = std::atol(argv[++i]);
    } else if (a == "-csv") {
      o.csv = true;
    } else if (a == "-q") {
      o.quiet = true;
    } else if (!a.empty() && a[0] == '-') {
      usage(argv[0]);
    } else {
      positional.push_back(a);
    }
  }
  if (positional.size() != 2) usage(argv[0]);
  o.ref = positional[0];
  o.test = positional[1];
  return o;
}

struct Result {
  std::size_t n = 0;
  std::size_t flagged = 0;
  std::size_t refNan = 0;
  std::size_t testNan = 0;
  std::size_t total = 0;
  double maxAbs = 0.0;
  double maxRe = 0.0;
  double maxIm = 0.0;
  double sumRe = 0.0;
  double sumIm = 0.0;
  double sumsqRe = 0.0;
  double sumsqIm = 0.0;
  double sumAbs = 0.0;
  double sigSumSq = 0.0;
  double maxRef = 0.0;
  std::vector<float> absErr;
  std::vector<double> chanSumsq;  // per (channel*pol) error power
  std::vector<std::size_t> chanN;
  std::vector<double> rowSumsq;  // per-row error power
  std::vector<std::size_t> rowN;
  // autocorrelation vs cross-correlation error power
  std::size_t autoN = 0, crossN = 0;
  double autoSumsq = 0.0, crossSumsq = 0.0;
  double autoMaxAbs = 0.0, crossMaxAbs = 0.0;
  double autoSigSumSq = 0.0, crossSigSumSq = 0.0;
  struct Worst {
    double ae;
    std::size_t row;
    std::size_t chpol;
    float refRe, refIm, testRe, testIm;
    int a1, a2;
  };
  std::vector<Worst> worst;
};

Result compareColumn(const std::string &refPath, const std::string &testPath,
                     const std::string &column, bool collectPercentiles,
                     bool collectPerChan, bool collectPerRow, long nWorst) {
  using Worst = Result::Worst;
  casacore::Table ref(refPath);
  casacore::Table test(testPath);

  if (!ref.tableDesc().isColumn(column) || !test.tableDesc().isColumn(column))
    throw std::runtime_error("column '" + column +
                             "' missing in one of the tables");
  if (ref.nrow() != test.nrow())
    throw std::runtime_error("row count differs: " +
                             std::to_string(ref.nrow()) + " vs " +
                             std::to_string(test.nrow()));

  casacore::ArrayColumn<casacore::Complex> refData(ref, column);
  casacore::ArrayColumn<casacore::Complex> testData(test, column);
  casacore::ArrayColumn<bool> refFlag(ref, "FLAG");
  casacore::ArrayColumn<bool> testFlag(test, "FLAG");
  casacore::ScalarColumn<int> ant1(ref, "ANTENNA1");
  casacore::ScalarColumn<int> ant2(ref, "ANTENNA2");
  const bool haveAnt = ref.tableDesc().isColumn("ANTENNA1") &&
                       ref.tableDesc().isColumn("ANTENNA2");

  Result r;
  const casacore::IPosition shape = refData.shape(0);
  const std::size_t rowSize = shape.product();
  if (static_cast<std::size_t>(testData.shape(0).product()) != rowSize)
    throw std::runtime_error("column shape differs");
  const std::size_t chPol = rowSize;  // flattened channel*pol dimension

  if (collectPerChan) {
    r.chanSumsq.assign(chPol, 0.0);
    r.chanN.assign(chPol, 0);
  }
  if (collectPercentiles)
    r.absErr.reserve(static_cast<std::size_t>(ref.nrow()) * rowSize);

  casacore::Array<casacore::Complex> refArr(shape);
  casacore::Array<casacore::Complex> testArr(shape);
  casacore::Array<bool> refFlagArr(shape);
  casacore::Array<bool> testFlagArr(shape);

  if (collectPerRow) {
    r.rowSumsq.assign(ref.nrow(), 0.0);
    r.rowN.assign(ref.nrow(), 0);
  }

  for (std::size_t row = 0; row != ref.nrow(); ++row) {
    refData.get(row, refArr);
    testData.get(row, testArr);
    refFlag.get(row, refFlagArr);
    testFlag.get(row, testFlagArr);

    const int a1 = haveAnt ? ant1(row) : -1;
    const int a2 = haveAnt ? ant2(row) : -1;
    auto rd = refArr.cbegin();
    auto td = testArr.cbegin();
    auto rf = refFlagArr.cbegin();
    auto tf = testFlagArr.cbegin();
    for (std::size_t i = 0; i != rowSize; ++i, ++rd, ++td, ++rf, ++tf) {
      ++r.total;
      if (*rf || *tf) {
        ++r.flagged;
        continue;
      }
      const float rr = rd->real(), ri = rd->imag();
      const float tr = td->real(), ti = td->imag();
      const bool refFin = std::isfinite(rr) && std::isfinite(ri);
      const bool testFin = std::isfinite(tr) && std::isfinite(ti);
      if (!refFin || !testFin) {
        if (!refFin) ++r.refNan;
        if (!testFin) ++r.testNan;
        continue;
      }
      const double dre = static_cast<double>(tr) - rr;
      const double dim = static_cast<double>(ti) - ri;
      const double are = std::fabs(dre);
      const double aim = std::fabs(dim);
      const double ae = std::sqrt(dre * dre + dim * dim);
      ++r.n;
      r.sumRe += dre;
      r.sumIm += dim;
      r.sumsqRe += dre * dre;
      r.sumsqIm += dim * dim;
      r.sumAbs += ae;
      if (are > r.maxRe) r.maxRe = are;
      if (aim > r.maxIm) r.maxIm = aim;
      if (ae > r.maxAbs) r.maxAbs = ae;
      const double sr = rr, si = ri;
      const double p = dre * dre + dim * dim;
      r.sigSumSq += sr * sr + si * si;
      if (a1 == a2) {
        ++r.autoN;
        r.autoSumsq += p;
        r.autoSigSumSq += sr * sr + si * si;
        if (ae > r.autoMaxAbs) r.autoMaxAbs = ae;
      } else {
        ++r.crossN;
        r.crossSumsq += p;
        r.crossSigSumSq += sr * sr + si * si;
        if (ae > r.crossMaxAbs) r.crossMaxAbs = ae;
      }
      const double m = std::max(std::fabs(sr), std::fabs(si));
      if (m > r.maxRef) r.maxRef = m;
      if (collectPercentiles) r.absErr.push_back(static_cast<float>(ae));
      if (collectPerChan) {
        r.chanSumsq[i] += p;
        r.chanN[i] += 1;
      }
      if (collectPerRow) {
        r.rowSumsq[row] += p;
        r.rowN[row] += 1;
      }
      if (nWorst > 0) {
        if (r.worst.size() < static_cast<std::size_t>(nWorst)) {
          r.worst.push_back({ae, row, i, rr, ri, tr, ti, a1, a2});
          std::push_heap(r.worst.begin(), r.worst.end(),
                         [](const Worst &a, const Worst &b) {
                           return a.ae > b.ae;
                         });
        } else if (ae > r.worst.front().ae) {
          std::pop_heap(r.worst.begin(), r.worst.end(),
                        [](const Worst &a, const Worst &b) {
                          return a.ae > b.ae;
                        });
          r.worst.back() = {ae, row, i, rr, ri, tr, ti, a1, a2};
          std::push_heap(r.worst.begin(), r.worst.end(),
                         [](const Worst &a, const Worst &b) {
                           return a.ae > b.ae;
                         });
        }
      }
    }
  }
  return r;
}

double percentile(std::vector<float> &v, double p) {
  if (v.empty()) return 0.0;
  const std::size_t idx = static_cast<std::size_t>(
      p * static_cast<double>(v.size() - 1) + 0.5);
  std::nth_element(v.begin(), v.begin() + idx, v.end());
  return v[idx];
}

void printResult(const std::string &column, Result &r) {
  const double n = static_cast<double>(r.n);
  const double mse = n > 0 ? (r.sumsqRe + r.sumsqIm) / n : 0.0;
  const double rmse = std::sqrt(mse);
  const double mseRe = n > 0 ? r.sumsqRe / n : 0.0;
  const double mseIm = n > 0 ? r.sumsqIm / n : 0.0;
  const double meanRe = n > 0 ? r.sumRe / n : 0.0;
  const double meanIm = n > 0 ? r.sumIm / n : 0.0;
  const double stdRe =
      n > 0 ? std::sqrt(std::max(0.0, mseRe - meanRe * meanRe)) : 0.0;
  const double stdIm =
      n > 0 ? std::sqrt(std::max(0.0, mseIm - meanIm * meanIm)) : 0.0;
  const double mae = n > 0 ? r.sumAbs / n : 0.0;
  const double sigPower = n > 0 ? r.sigSumSq / n : 0.0;
  const double sigRms = std::sqrt(sigPower);
  const double relRmse = sigRms > 0 ? rmse / sigRms : 0.0;
  const double snr = mse > 0 ? 10.0 * std::log10(sigPower / mse) : 0.0;

  std::printf("  column          : %s\n", column.c_str());
  std::printf("  samples total   : %zu  (flagged %zu = %.3f%%)\n", r.total,
              r.flagged, r.total ? 100.0 * r.flagged / r.total : 0.0);
  std::printf("  compared        : %zu  (unflagged & finite; %.4f%% of "
              "unflagged)\n",
              r.n,
              (r.total - r.flagged)
                  ? 100.0 * r.n / static_cast<double>(r.total - r.flagged)
                  : 0.0);
  std::printf("  non-finite      : ref %zu  test %zu\n", r.refNan, r.testNan);
  std::printf("  max |diff|      : %.6g   (|re| %.6g, |im| %.6g)\n", r.maxAbs,
              r.maxRe, r.maxIm);
  std::printf("  mean err re/im  : %.6g  %.6g\n", meanRe, meanIm);
  std::printf("  std  err re/im  : %.6g  %.6g\n", stdRe, stdIm);
  std::printf("  MSE  re/im      : %.6g  %.6g\n", mseRe, mseIm);
  std::printf("  MSE (complex)   : %.6g\n", mse);
  std::printf("  RMSE            : %.6g\n", rmse);
  std::printf("  MAE             : %.6g\n", mae);
  std::printf("  RMS signal      : %.6g   (max |ref| %.6g)\n", sigRms,
              r.maxRef);
  std::printf("  relative RMSE   : %.6g  (%.4f%%)\n", relRmse,
              100.0 * relRmse);
  std::printf("  SNR             : %.3f dB\n", snr);
  if (!r.absErr.empty()) {
    std::printf("  |diff| p50/p90/p99/p99.9 : %.6g / %.6g / %.6g / %.6g\n",
                percentile(r.absErr, 0.50), percentile(r.absErr, 0.90),
                percentile(r.absErr, 0.99), percentile(r.absErr, 0.999));
  }
  if (r.autoN || r.crossN) {
    const double an = static_cast<double>(r.autoN);
    const double cn = static_cast<double>(r.crossN);
    std::printf("  auto  : n=%zu  rmse=%.6g  relRmse=%.4f%%  max|diff|=%.6g\n",
                r.autoN, an ? std::sqrt(r.autoSumsq / an) : 0.0,
                an ? 100.0 * std::sqrt((r.autoSumsq / an) /
                                       (r.autoSigSumSq / an))
                   : 0.0,
                r.autoMaxAbs);
    std::printf("  cross : n=%zu  rmse=%.6g  relRmse=%.4f%%  max|diff|=%.6g\n",
                r.crossN, cn ? std::sqrt(r.crossSumsq / cn) : 0.0,
                cn ? 100.0 * std::sqrt((r.crossSumsq / cn) /
                                       (r.crossSigSumSq / cn))
                   : 0.0,
                r.crossMaxAbs);
  }
}

void printCsv(const std::string &tag, const std::string &column,
              const Result &r) {
  const double n = static_cast<double>(r.n);
  const double mse = n > 0 ? (r.sumsqRe + r.sumsqIm) / n : 0.0;
  const double rmse = std::sqrt(mse);
  const double sigPower = n > 0 ? r.sigSumSq / n : 0.0;
  const double sigRms = std::sqrt(sigPower);
  const double relRmse = sigRms > 0 ? rmse / sigRms : 0.0;
  const double snr = mse > 0 ? 10.0 * std::log10(sigPower / mse) : 0.0;
  std::printf(
      "%s,%s,%zu,%zu,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
              tag.c_str(), column.c_str(), r.n, r.flagged, r.maxAbs, r.maxRe,
              r.maxIm, n ? r.sumRe / n : 0.0, n ? r.sumIm / n : 0.0, mse, rmse,
              sigRms, relRmse, snr);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options o = parse(argc, argv);
    if (o.csv)
      std::printf("tag,column,n,flagged,maxAbsErr,maxErrRe,maxErrIm,"
                  "meanErrRe,meanErrIm,mse,rmse,sigRms,relRmse,snrDb\n");
    for (const std::string &col : o.columns) {
      Result r = compareColumn(o.ref, o.test, col, !o.csv, o.perChannel,
                               o.perRow, o.worst);
      if (o.csv) {
        printCsv(o.test, col, r);
      } else {
        if (!o.quiet)
          std::printf("\n== %s : %s ==\n", o.test.c_str(), col.c_str());
        printResult(col, r);
        if (o.worst > 0 && !r.worst.empty()) {
          std::sort(r.worst.begin(), r.worst.end(),
                    [](const Result::Worst &a, const Result::Worst &b) {
                      return a.ae > b.ae;
                    });
          std::printf("  worst |diff| samples (row, ch*pol, refRe, refIm, "
                      "testRe, testIm):\n");
          for (const auto &w : r.worst)
            std::printf("    %.6g  row=%zu cp=%zu a1=%d a2=%d  ref=(%.6g,%.6g) "
                        "test=(%.6g,%.6g)\n",
                        w.ae, w.row, w.chpol, w.a1, w.a2, w.refRe, w.refIm,
                        w.testRe, w.testIm);
        }
        if (o.perRow && !r.rowSumsq.empty()) {
          std::vector<std::size_t> idx(r.rowSumsq.size());
          for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
          std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            return r.rowSumsq[a] > r.rowSumsq[b];
          });
          std::printf("  worst rows by error power:\n");
          for (std::size_t k = 0; k < idx.size() && k < 20; ++k) {
            const std::size_t rw = idx[k];
            const double rms = r.rowN[rw]
                                   ? std::sqrt(r.rowSumsq[rw] / r.rowN[rw])
                                   : 0.0;
            std::printf("    row %zu  n=%zu  rmse=%.6g\n", rw, r.rowN[rw],
                        rms);
          }
        }
        if (o.perChannel && !r.chanSumsq.empty()) {
          std::printf("  per-(ch,pol) RMSE:\n");
          for (std::size_t i = 0; i < r.chanSumsq.size(); ++i) {
            const double v =
                r.chanN[i] ? std::sqrt(r.chanSumsq[i] / r.chanN[i]) : 0.0;
            std::printf(" %zu:%.4g", i, v);
            if ((i + 1) % 12 == 0) std::printf("\n");
          }
          std::printf("\n");
        }
      }
    }
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "mscompare: %s\n", e.what());
    return 1;
  }
}
