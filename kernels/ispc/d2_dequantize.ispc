// D2 (ISPC) — dequantization / decoding.
//
// ISPC port of `kernels/cuda/d2_dequantize.cu`.
//
//   value = decValues[symbol] * channelRms[ch,pol] * g_a1 * g_a2
//
// with the NaN symbol (index == decCount) producing a NaN visibility, exactly
// like the CPU `AFTimeBlockEncoder::Decode`.  The output is the interleaved
// AoS layout; the two writes per visibility are contiguous, and the metadata
// lookups (channel factor and the two antenna gains) are broadcast/gather from
// tiny arrays.

#include "ispc_common.isph"

export void ispc_d2_dequantize(
    const uniform uint32 symbols[], const uniform float meta[],
    const uniform int ant1[], const uniform int ant2[],
    const uniform float decValues[], uniform float interleaved[],
    uniform int nRows, uniform int visPerRow, uniform int nPol,
    uniform int nAnt, uniform int decCount, uniform double nanValue) {
  const uniform int nVis = nRows * visPerRow;
  foreach (i = 0 ... nVis) {
    const int row = i / visPerRow;
    const int rem = i - row * visPerRow;
    const int ch = rem / nPol;
    const int pol = rem - ch * nPol;
    const double chRms = (double)meta[ch * nPol + pol];
    const double antFac = (double)meta[visPerRow + pol * nAnt + ant1[row]] *
                          (double)meta[visPerRow + pol * nAnt + ant2[row]];
    const double factor = chRms * antFac;
    const uint32 sRe = symbols[2 * i];
    const uint32 sIm = symbols[2 * i + 1];
    const double vRe =
        (sRe < (uint32)decCount) ? (double)decValues[sRe] : nanValue;
    const double vIm =
        (sIm < (uint32)decCount) ? (double)decValues[sIm] : nanValue;
    interleaved[2 * i] = (float)(vRe * factor);
    interleaved[2 * i + 1] = (float)(vIm * factor);
  }
}
