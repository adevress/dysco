// K7 (ISPC) — dithered dictionary quantization.
//
// ISPC port of `kernels/cuda/k7_quantize_dither.cu`.  Every value is
// quantized with a counter-based Philox dither indexed by the global symbol
// address, so the output is independent of lane assignment and identical to
// the CPU/CUDA backends.  The binary search over the decodes is a gather, but
// the dictionary is tiny and stays hot in cache, so it is cheap.

#include "ispc_common.isph"

// Flat variant used by the unit tests: one symbol per input value.
export void ispc_k7_quantize_flat(
    const uniform float values[], const uniform float decValues[],
    uniform int decCount, uniform uint32 symbols[], uniform int n,
    uniform uint32 seedLo, uniform uint32 seedHi, uniform uint64 blockIndex) {
  foreach (i = 0 ... n) {
    const uint32 dither =
        ispc_philox_dither(seedLo, seedHi, blockIndex, (uint64)i);
    symbols[i] = ispc_quantize_dither(values[i], decValues, decCount, dither);
  }
}

// AF path: the real and imaginary parts of visibility `i` use symbol indices
// 2*i and 2*i+1.
export void ispc_k7_quantize_dither(
    const uniform double re[], const uniform double im[],
    const uniform float decValues[], uniform int decCount,
    uniform uint32 symbols[], uniform int nVis, uniform uint32 seedLo,
    uniform uint32 seedHi, uniform uint64 blockIndex) {
  foreach (i = 0 ... nVis) {
    const uint32 dRe =
        ispc_dither_for_visibility(seedLo, seedHi, blockIndex, (uint64)i, 0);
    const uint32 dIm =
        ispc_dither_for_visibility(seedLo, seedHi, blockIndex, (uint64)i, 1);
    symbols[2 * i] =
        ispc_quantize_dither((float)re[i], decValues, decCount, dRe);
    symbols[2 * i + 1] =
        ispc_quantize_dither((float)im[i], decValues, decCount, dIm);
  }
}
