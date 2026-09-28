// D1 — inverse of K8, bit unpacking.

#include "cuda_kernels.h"

namespace dyscostman {
namespace gpu {

namespace {

__global__ void unpack8Kernel(const unsigned char *packed, size_t symbolCount,
                              unsigned int *symbols) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= symbolCount) return;
  symbols[i] = packed[i];
}

__global__ void unpackGenericKernel(const unsigned char *packed, unsigned bits,
                                    unsigned int *symbols, size_t symbolCount) {
  const size_t symIdx = static_cast<size_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (symIdx >= symbolCount) return;
  const size_t bit0 = symIdx * bits;
  unsigned v = 0;
  for (unsigned k = 0; k != bits; ++k) {
    const size_t bit = bit0 + k;
    v |= ((packed[bit >> 3] >> (bit & 7)) & 1u) << k;
  }
  symbols[symIdx] = v;
}

}  // namespace

void d1_bit_unpack(dysco::span<const unsigned char> packed, unsigned bits,
                   dysco::span<unsigned int> symbols, dysco_stream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t symbolCount = symbols.size();
  const auto blocks =
      static_cast<unsigned>((symbolCount + kThreads - 1) / kThreads);
  if (bits == 8)
    DYSCO_LAUNCH_1D(unpack8Kernel, blocks, kThreads, stream, packed.data(),
                    symbolCount, symbols.data());
  else
    DYSCO_LAUNCH_1D(unpackGenericKernel, blocks, kThreads, stream, packed.data(),
                    bits, symbols.data(), symbolCount);
}

}  // namespace gpu
}  // namespace dyscostman
