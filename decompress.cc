#include "dyscostman.h"
#include "stmanmodifier.h"

#include <iostream>
#include <string>

using namespace dyscostman;

int main(int argc, char *argv[]) {
  register_dyscostman();

  bool useGpu = false;
  std::string msPath;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "-cuda")
      useGpu = true;
    else
      msPath = arg;
  }
  if (msPath.empty()) {
    std::cerr << "Usage: decompress [-cuda] <ms>\n"
                 "  -cuda   decode using the CUDA backend (requires "
                 "ENABLE_CUDA)\n";
    return 0;
  }
  if (useGpu) {
    std::cerr << "Enabling CUDA backend for decoding...\n";
    SetGlobalGpuEnabled(true);
  }
  std::unique_ptr<casacore::Table> ms(
      new casacore::Table(msPath, casacore::Table::Update));
  StManModifier modifier(*ms);

  bool isDataReplaced = modifier.InitColumnWithDefaultStMan("DATA", false);
  bool isWeightReplaced =
      modifier.InitColumnWithDefaultStMan("WEIGHT_SPECTRUM", true);

  if (isDataReplaced) modifier.MoveColumnData<casacore::Complex>("DATA");
  if (isWeightReplaced) modifier.MoveColumnData<float>("WEIGHT_SPECTRUM");

  if (isDataReplaced || isWeightReplaced) {
    StManModifier::Reorder(ms, msPath);
  }

  std::cout << "Finished.\n";
}
