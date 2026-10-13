#include "dyscostman.h"
#include "stmanmodifier.h"

#include <iostream>
#include <memory>
#include <string>

using namespace dyscostman;

int main(int argc, char *argv[]) {
  register_dyscostman();

  bool useGpu = false;
  bool useIspc = false;
  std::string msPath;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-cuda")
      useGpu = true;
    else if (arg == "-ispc")
      useIspc = true;
    else
      msPath = arg;
  }
  if (msPath.empty()) {
    std::cerr << "Usage: decompress [-cuda] [-ispc] <ms>\n"
                 "  -cuda   decode using the CUDA backend (requires "
                 "ENABLE_CUDA)\n"
                 "  -ispc   decode using the ISPC backend (requires "
                 "ENABLE_ISPC)\n";
    return 0;
  }
  if (useGpu) {
    std::cerr << "Enabling CUDA backend for decoding...\n";
    SetGlobalGpuEnabled(true);
  } else if (useIspc) {
    std::cerr << "Enabling ISPC backend for decoding...\n";
    SetGlobalIspcEnabled(true);
  }
  std::unique_ptr<casacore::Table> ms(
      new casacore::Table(msPath, casacore::Table::Update));
  StManModifier modifier(*ms);

  bool isDataReplaced = modifier.InitColumnWithDefaultStMan("DATA", false);
  // WEIGHT_SPECTRUM is optional: many measurement sets (e.g. SKA Low) do not
  // have one, so only decompress it when present.
  bool isWeightReplaced = false;
  if (ms->tableDesc().isColumn("WEIGHT_SPECTRUM"))
    isWeightReplaced =
        modifier.InitColumnWithDefaultStMan("WEIGHT_SPECTRUM", true);

  if (isDataReplaced) modifier.MoveColumnData<casacore::Complex>("DATA");
  if (isWeightReplaced) modifier.MoveColumnData<float>("WEIGHT_SPECTRUM");

  if (isDataReplaced || isWeightReplaced) {
    StManModifier::Reorder(ms, msPath);
  }

  std::cout << "Finished.\n";
}
