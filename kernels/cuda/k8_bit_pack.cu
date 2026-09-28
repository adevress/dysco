// K8 — bit packing of the quantization symbols.
//
// The packed representation is a little-endian bit stream: symbol 0 occupies
// the lowest bits of byte 0, the next symbol the following bits, and values
// cross byte boundaries LSB-first.  This is exactly the convention used by
// BytePacker (pack2/pack3/... in bytepacker.h).  An 8-bit fast path is
// provided because it is the default Dysco bit rate.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void pack8Kernel(const unsigned int *symbols, size_t symbolCount,
                            unsigned char *packed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= symbolCount) return;
  packed[i] = static_cast<unsigned char>(symbols[i]);
}

__global__ void packGenericKernel(const unsigned int *symbols,
                                  size_t symbolCount, unsigned bits,
                                  unsigned char *packed, size_t packedBytes) {
  const size_t b = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= packedBytes) return;
  unsigned out = 0;
  for (unsigned k = 0; k != 8; ++k) {
    const size_t bit = b * 8 + k;
    const size_t symIdx = bit / bits;
    if (symIdx < symbolCount) {
      const unsigned bitInSym = static_cast<unsigned>(bit % bits);
      out |= ((symbols[symIdx] >> bitInSym) & 1u) << k;
    }
  }
  packed[b] = static_cast<unsigned char>(out);
}

}  // namespace

void k8_bit_pack(const unsigned int *symbols, size_t symbolCount, unsigned bits,
                 unsigned char *packed, size_t packedBytes,
                 dysco_stream_t stream) {
  const unsigned threads = 256;
  if (bits == 8) {
    const unsigned blocks =
        static_cast<unsigned>((symbolCount + threads - 1) / threads);
    DYSCO_LAUNCH_1D(pack8Kernel, blocks, threads, stream, symbols, symbolCount,
                    packed);
  } else {
    const unsigned blocks =
        static_cast<unsigned>((packedBytes + threads - 1) / threads);
    DYSCO_LAUNCH_1D(packGenericKernel, blocks, threads, stream, symbols,
                    symbolCount, bits, packed, packedBytes);
  }
}

}  // namespace gpu
}  // namespace dyscostman
