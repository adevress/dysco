#include <boost/test/unit_test.hpp>
#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/directory.hpp>

#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>

#include "../dyscostman.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using namespace casacore;
using namespace dyscostman;

BOOST_AUTO_TEST_SUITE(dyscostman)

casacore::Record GetDyscoSpec(bool staticSeed = false) {
  casacore::Record dyscoSpec;
  dyscoSpec.define("distribution", "TruncatedGaussian");
  dyscoSpec.define("normalization", "AF");
  dyscoSpec.define("distributionTruncation", 2.0);
  dyscoSpec.define("dataBitCount", 10);
  dyscoSpec.define("weightBitCount", 12);
  dyscoSpec.define("staticSeed", staticSeed);
  return dyscoSpec;
}

struct TestTableFixture {
  explicit TestTableFixture(size_t nAnt, const std::string &dir = "TestTable",
                            bool staticSeed = false, size_t nTimeSteps = 2)
      : _dir(dir) {
    casacore::TableDesc tableDesc;
    IPosition shape(2, 1, 1);
    casacore::ArrayColumnDesc<casacore::Complex> columnDesc(
        "DATA", "", "DyscoStMan", "", shape);
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
    casacore::SetupNewTable setupNewTable(dir, tableDesc, casacore::Table::New);

    register_dyscostman();
    DataManagerCtor dyscoConstructor = DataManager::getCtor("DyscoStMan");
    std::unique_ptr<DataManager> dysco(
        dyscoConstructor("DATA_dm", GetDyscoSpec(staticSeed)));
    setupNewTable.bindColumn("DATA", *dysco);
    casacore::Table newTable(setupNewTable);

    size_t a1 = 0, a2 = 1;
    double time = 10.0;
    const size_t nRow = nTimeSteps * nAnt * (nAnt - 1) / 2;
    newTable.addRow(nRow);
    casacore::ScalarColumn<int> a1Col(newTable, "ANTENNA1"),
        a2Col(newTable, "ANTENNA2"), fieldCol(newTable, "FIELD_ID"),
        dataDescIdCol(newTable, "DATA_DESC_ID");
    casacore::ScalarColumn<double> timeCol(newTable, "TIME");
    for (size_t i = 0; i != nRow; ++i) {
      a1Col.put(i, a1);
      a2Col.put(i, a2);
      fieldCol.put(i, 0);
      dataDescIdCol.put(i, 0);
      timeCol.put(i, time);
      a2++;
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
    for (size_t i = 0; i != nRow; ++i) {
      casacore::Array<casacore::Complex> arr(shape);
      *arr.cbegin() = i;
      dataCol.put(i, arr);
    }
  }
  ~TestTableFixture() { boost::filesystem::remove_all(_dir); }

  std::string _dir;
};

BOOST_AUTO_TEST_CASE(spec) {
  DyscoStMan dysco(8, 12);
  Record spec = dysco.dataManagerSpec();
  BOOST_CHECK_EQUAL(spec.asInt("dataBitCount"), 8);
  BOOST_CHECK_EQUAL(spec.asInt("weightBitCount"), 12);
}

BOOST_AUTO_TEST_CASE(name) {
  DyscoStMan dysco1(8, 12, "withparameters");
  BOOST_CHECK_EQUAL(dysco1.dataManagerType(), "DyscoStMan");
  BOOST_CHECK_EQUAL(dysco1.dataManagerName(), "withparameters");

  DyscoStMan dysco2("testname", GetDyscoSpec());
  BOOST_CHECK_EQUAL(dysco2.dataManagerType(), "DyscoStMan");
  BOOST_CHECK_EQUAL(dysco2.dataManagerName(), "testname");

  std::unique_ptr<DataManager> dysco3(dysco2.clone());
  BOOST_CHECK_EQUAL(dysco3->dataManagerType(), "DyscoStMan");
  BOOST_CHECK_EQUAL(dysco3->dataManagerName(), "testname");

  register_dyscostman();
  DataManagerCtor dyscoConstructor = DataManager::getCtor("DyscoStMan");
  std::unique_ptr<DataManager> dysco4(
      dyscoConstructor("Constructed", GetDyscoSpec()));
  BOOST_CHECK_EQUAL(dysco4->dataManagerName(), "Constructed");

  TestTableFixture fixture(3);
  casacore::Table table("TestTable");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  DataManager* dm = table.findDataManager("DATA", true);
  BOOST_CHECK_EQUAL(dm->dataManagerName(), "DATA_dm");
}

BOOST_AUTO_TEST_CASE(makecolumn) {
  DyscoStMan dysco(8, 12, "mydysco");
  dysco.createDirArrColumn("DATA", casacore::DataType::TpComplex, "");
  dysco.createDirArrColumn("WEIGHT_SPECTRUM", casacore::DataType::TpFloat, "");
  dysco.createDirArrColumn("CORRECTED_DATA", casacore::DataType::TpComplex, "");
  dysco.createDirArrColumn("ANYTHING", casacore::DataType::TpComplex, "");
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(maketable) {
  size_t nAnt = 3;
  TestTableFixture fixture(nAnt);

  casacore::Table table("TestTable");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  for (size_t i = 0; i != table.nrow(); ++i) {
    BOOST_CHECK_CLOSE_FRACTION((*dataCol(i).cbegin()).real(), float(i), 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(read_past_end, * boost::unit_test::disabled()) {
  /**
   * While reading past the end of a file might seem wrong in any case, it can
   * happen that a user reads a line that was not stored yet in the particular
   * column, but which does exist in the table. DP3 may do this.
   *
   * However, when Casacore is compiled in debug mode, an assert fails when
   * trying to write past the end of the table -- hence this test is disabled.
   */
  size_t nAnt = 3;
  TestTableFixture fixture(nAnt);

  casacore::Table table("TestTable");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  for (size_t i = table.nrow(); i != table.nrow() * 2; ++i) {
    BOOST_CHECK_CLOSE_FRACTION((*dataCol(i).cbegin()).real(), 0.0, 1e-4);
  }
}

BOOST_AUTO_TEST_CASE(readonly) {
  size_t nAnt = 3;
  TestTableFixture fixture(nAnt);

  boost::filesystem::directory_iterator end_itr;

  for (boost::filesystem::directory_iterator itr("TestTable/"); itr != end_itr;
       ++itr) {
    if (boost::filesystem::is_regular_file(itr->path())) {
      boost::filesystem::permissions(
          itr->path(),
          boost::filesystem::others_read | boost::filesystem::owner_read);
    }
  }
  casacore::Table table("TestTable");
  casacore::ArrayColumn<casacore::Complex> dataCol(table, "DATA");
  for (size_t i = 0; i != table.nrow(); ++i) {
    BOOST_CHECK_CLOSE_FRACTION((*dataCol(i).cbegin()).real(), float(i), 1e-4);
  }
}

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  std::ostringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

}  // namespace

// The compressed bytes must not depend on how many threads encode the blocks:
// the counter-based dither is a pure function of the symbol's logical address,
// so thread count and scheduling are irrelevant.  This is a regression test for
// the removed std::mt19937 path, whose output depended on thread scheduling.
BOOST_AUTO_TEST_CASE(thread_count_invariance) {
  const size_t nAnt = 8, nTimeSteps = 8;

  setenv("DYSCO_THREAD_COUNT", "1", 1);
  TestTableFixture fixture1(nAnt, "ThreadTable1", /*staticSeed=*/true,
                            nTimeSteps);
  setenv("DYSCO_THREAD_COUNT", "4", 1);
  TestTableFixture fixture4(nAnt, "ThreadTable4", /*staticSeed=*/true,
                            nTimeSteps);
  unsetenv("DYSCO_THREAD_COUNT");

  // `table.f0` is the file of the first storage manager, i.e. the Dysco
  // compressed column.  `table.f1` belongs to the default manager that holds
  // the scalar (antenna/time) columns and is not compared here.
  const std::string file1 = "ThreadTable1/table.f0";
  const std::string file4 = "ThreadTable4/table.f0";
  BOOST_REQUIRE_MESSAGE(boost::filesystem::exists(file1),
                        "missing storage manager file " << file1);
  BOOST_REQUIRE_MESSAGE(boost::filesystem::exists(file4),
                        "missing storage manager file " << file4);

  const std::string bytes1 = ReadFile(file1);
  const std::string bytes4 = ReadFile(file4);
  BOOST_CHECK_EQUAL(bytes1.size(), bytes4.size());
  BOOST_CHECK_MESSAGE(bytes1 == bytes4,
                      "compressed output differs between 1 and 4 threads");
}

BOOST_AUTO_TEST_SUITE_END()
