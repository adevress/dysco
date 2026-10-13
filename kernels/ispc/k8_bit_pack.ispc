// K8 (ISPC) — bit packing of the quantization symbols.
//
// ISPC port of `kernels/cuda/k8_bit_pack.cu`.  The packed representation is a
// little-endian bit stream matching BytePacker.  One work-item per output byte
// keeps the writes contiguous and race-free; the reads of the source symbols
// are gathers, which is unavoidable for a sub-byte bit stream.  The 8-bit fast
// path (the default Dysco bit rate) is a pure contiguous copy.

#include "ispc_common.isph"

export void ispc_k8_bit_pack(const uniform uint32 symbols[],
                             uniform int symbolCount, uniform uint32 bits,
                             uniform uint8 packed[], uniform int packedBytes) {
  if (bits == 8) {
    foreach (i = 0 ... symbolCount) packed[i] = (uint8)symbols[i];
    return;
  }
  foreach (b = 0 ... packedBytes) {
    uint32 out = 0;
    for (uint32 k = 0; k < 8; ++k) {
      const uint64 bit = (uint64)b * 8 + k;
      const uint64 symIdx = bit / bits;
      if (symIdx < (uint64)symbolCount) {
        const uint32 bitInSym = (uint32)(bit % bits);
        out |= ((symbols[symIdx] >> bitInSym) & 1u) << k;
      }
    }
    packed[b] = (uint8)out;
  }
}
