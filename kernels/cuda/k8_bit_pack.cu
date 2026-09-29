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

__global__ void pack8Kernel(cuda::std::span<const unsigned int> symbols,
                            cuda::std::span<unsigned char> packed) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= symbols.size()) return;
  packed[i] = static_cast<unsigned char>(symbols[i]);
}

__global__ void packGenericKernel(cuda::std::span<const unsigned int> symbols,
                                  unsigned bits,
                                  cuda::std::span<unsigned char> packed) {
  const std::size_t b =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (b >= packed.size()) return;
  const std::size_t symbolCount = symbols.size();
  unsigned out = 0;
  for (unsigned k = 0; k != 8; ++k) {
    const std::size_t bit = b * 8 + k;
    const std::size_t symIdx = bit / bits;
    if (symIdx < symbolCount) {
      const unsigned bitInSym = static_cast<unsigned>(bit % bits);
      out |= ((symbols[symIdx] >> bitInSym) & 1u) << k;
    }
  }
  packed[b] = static_cast<unsigned char>(out);
}

}  // namespace

void k8_bit_pack(cuda::std::span<const unsigned int> symbols, unsigned bits,
                 cuda::std::span<unsigned char> packed, dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  if (bits == 8) {
    const auto blocks =
        static_cast<unsigned>((symbols.size() + kThreads - 1) / kThreads);
    DYSCO_LAUNCH_1D(pack8Kernel, blocks, kThreads, stream, symbols, packed);
  } else {
    const auto blocks =
        static_cast<unsigned>((packed.size() + kThreads - 1) / kThreads);
    DYSCO_LAUNCH_1D(packGenericKernel, blocks, kThreads, stream, symbols, bits,
                    packed);
  }
}

}  // namespace gpu
}  // namespace dyscostman
