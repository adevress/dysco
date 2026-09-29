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

__global__ void pack8Kernel(cuda::std::span<const std::uint32_t> symbols,
                            cuda::std::span<std::uint8_t> packed) {
  const std::size_t i = threadFlattenedId();
  if (i >= symbols.size()) return;
  packed[i] = static_cast<std::uint8_t>(symbols[i]);
}

__global__ void packGenericKernel(cuda::std::span<const std::uint32_t> symbols,
                                  std::uint32_t bits,
                                  cuda::std::span<std::uint8_t> packed) {
  const std::size_t b = threadFlattenedId();
  if (b >= packed.size()) return;
  const std::size_t symbolCount = symbols.size();
  std::uint32_t out = 0;
  for (std::uint32_t k = 0; k != 8; ++k) {
    const std::size_t bit = b * 8 + k;
    const std::size_t symIdx = bit / bits;
    if (symIdx < symbolCount) {
      const std::uint32_t bitInSym = static_cast<std::uint32_t>(bit % bits);
      out |= ((symbols[symIdx] >> bitInSym) & 1u) << k;
    }
  }
  packed[b] = static_cast<std::uint8_t>(out);
}

}  // namespace

void k8_bit_pack(cuda::std::span<const std::uint32_t> symbols,
                 std::uint32_t bits, cuda::std::span<std::uint8_t> packed,
                 dysco_stream_t stream) {
  constexpr std::uint32_t kThreads = kThreadBlockSize;
  if (bits == 8) {
    const auto blocks = gridFor(symbols.size(), kThreads);
    DYSCO_LAUNCH_1D(pack8Kernel, blocks, kThreads, stream, symbols, packed);
  } else {
    const auto blocks = gridFor(packed.size(), kThreads);
    DYSCO_LAUNCH_1D(packGenericKernel, blocks, kThreads, stream, symbols, bits,
                    packed);
  }
}

}  // namespace gpu
}  // namespace dyscostman
