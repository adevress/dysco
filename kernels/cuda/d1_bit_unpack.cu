// D1 — inverse of K8, bit unpacking.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void unpack8Kernel(cuda::std::span<const unsigned char> packed,
                              cuda::std::span<unsigned int> symbols) {
  const std::size_t i = threadFlattenedId();
  if (i >= symbols.size()) return;
  symbols[i] = packed[i];
}

__global__ void unpackGenericKernel(cuda::std::span<const unsigned char> packed,
                                    unsigned bits,
                                    cuda::std::span<unsigned int> symbols) {
  const std::size_t symIdx = threadFlattenedId();
  if (symIdx >= symbols.size()) return;
  const std::size_t bit0 = symIdx * bits;
  unsigned v = 0;
  for (unsigned k = 0; k != bits; ++k) {
    const std::size_t bit = bit0 + k;
    v |= ((packed[bit >> 3] >> (bit & 7)) & 1u) << k;
  }
  symbols[symIdx] = v;
}

}  // namespace

void d1_bit_unpack(cuda::std::span<const unsigned char> packed, unsigned bits,
                   cuda::std::span<unsigned int> symbols,
                   dysco_stream_t stream) {
  constexpr unsigned kThreads = kThreadBlockSize;
  const auto blocks = gridFor(symbols.size(), kThreads);
  if (bits == 8)
    DYSCO_LAUNCH_1D(unpack8Kernel, blocks, kThreads, stream, packed, symbols);
  else
    DYSCO_LAUNCH_1D(unpackGenericKernel, blocks, kThreads, stream, packed, bits,
                    symbols);
}

}  // namespace gpu
}  // namespace dyscostman
