// D1 — inverse of K8, bit unpacking.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void unpack8Kernel(cuda::std::span<const unsigned char> packed,
                              cuda::std::span<unsigned int> symbols) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= symbols.size()) return;
  symbols[i] = packed[i];
}

__global__ void unpackGenericKernel(cuda::std::span<const unsigned char> packed,
                                    unsigned bits,
                                    cuda::std::span<unsigned int> symbols) {
  const std::size_t symIdx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
  constexpr unsigned kThreads = 256;
  const auto blocks =
      static_cast<unsigned>((symbols.size() + kThreads - 1) / kThreads);
  if (bits == 8)
    DYSCO_LAUNCH_1D(unpack8Kernel, blocks, kThreads, stream, packed, symbols);
  else
    DYSCO_LAUNCH_1D(unpackGenericKernel, blocks, kThreads, stream, packed, bits,
                    symbols);
}

}  // namespace gpu
}  // namespace dyscostman
