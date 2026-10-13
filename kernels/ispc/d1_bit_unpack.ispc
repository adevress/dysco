// D1 (ISPC) — inverse of K8, bit unpacking.
//
// ISPC port of `kernels/cuda/d1_bit_unpack.cu`.  One work-item per symbol; the
// byte reads are gathers across the packed stream, which is inherent to
// sub-byte unpacking.  The 8-bit fast path is a contiguous copy.

#include "ispc_common.isph"

export void ispc_d1_bit_unpack(const uniform uint8 packed[], uniform uint32 bits,
                               uniform uint32 symbols[], uniform int symbolCount,
                               uniform int packedBytes) {
  if (bits == 8) {
    foreach (i = 0 ... symbolCount) symbols[i] = (uint32)packed[i];
    return;
  }
  foreach (symIdx = 0 ... symbolCount) {
    const uint64 bit0 = (uint64)symIdx * bits;
    uint32 v = 0;
    for (uint32 k = 0; k < bits; ++k) {
      const uint64 bit = bit0 + k;
      v |= ((uint32)((packed[bit >> 3] >> (bit & 7)) & 1u)) << k;
    }
    symbols[symIdx] = v;
  }
}
