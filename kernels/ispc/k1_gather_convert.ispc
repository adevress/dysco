// K1 (ISPC) — gather + layout transform + float->double widening.
//
// ISPC port of `kernels/cuda/k1_gather_convert.cu`.  The input is the
// AoS (re,im,re,im,...) row-major visibility block and the output is the
// SoA double-precision `re[]`/`im[]` layout used by the rest of the pipeline.
// The stride-2 read is a gather; it is inherent to the AoS input and the
// layout transform is exactly what buys the following kernels contiguous,
// vector-friendly SoA access.

#include "ispc_common.isph"

export void ispc_k1_gather_convert(const uniform float interleaved[],
                                   uniform double re[], uniform double im[],
                                   uniform int nVis) {
  foreach (i = 0 ... nVis) {
    re[i] = (double)interleaved[2 * i];
    im[i] = (double)interleaved[2 * i + 1];
  }
}
