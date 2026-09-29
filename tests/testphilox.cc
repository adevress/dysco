#include "../kernels/philox_rng.h"

#include <boost/test/unit_test.hpp>

#include <cstdint>

using namespace dyscostman::random123;

// Host-only known-answer tests for the counter-based Philox4x32-10 generator.
// The expected values are the official Random123 `kat_vectors` for
// `philox4x32 10` (tests/kat_vectors of DEShawResearch/random123).  Running
// these in the plain (non-CUDA) build guarantees that the CPU Philox path is
// the standard algorithm, independent of the CUDA backend.

BOOST_AUTO_TEST_SUITE(philox)

namespace {

void CheckPhilox(std::uint32_t c0, std::uint32_t c1, std::uint32_t c2,
                 std::uint32_t c3, std::uint32_t k0, std::uint32_t k1,
                 std::uint32_t e0, std::uint32_t e1, std::uint32_t e2,
                 std::uint32_t e3) {
  philox4x32_10(c0, c1, c2, c3, k0, k1);
  BOOST_CHECK_EQUAL(c0, e0);
  BOOST_CHECK_EQUAL(c1, e1);
  BOOST_CHECK_EQUAL(c2, e2);
  BOOST_CHECK_EQUAL(c3, e3);
}

}  // namespace

// philox4x32 10 | ctr: 00000000 00000000 00000000 00000000
//                  key: 00000000 00000000
//                  out: 6627e8d5 e169c58d bc57ac4c 9b00dbd8
BOOST_AUTO_TEST_CASE(kat_all_zero) {
  CheckPhilox(0, 0, 0, 0, 0, 0, 0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu,
              0x9b00dbd8u);
}

// philox4x32 10 | ctr: ffffffff ffffffff ffffffff ffffffff
//                  key: ffffffff ffffffff
//                  out: 408f276d 41c83b0e a20bc7c6 6d5451fd
BOOST_AUTO_TEST_CASE(kat_all_ones) {
  CheckPhilox(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu,
              0xffffffffu, 0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu);
}

// philox4x32 10 | ctr: 243f6a88 85a308d3 13198a2e 03707344
//                  key: a4093822 299f31d0
//                  out: d16cfe09 94fdcceb 5001e420 24126ea1
BOOST_AUTO_TEST_CASE(kat_pi_e) {
  CheckPhilox(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u, 0xa4093822u,
              0x299f31d0u, 0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u);
}

// The dither domain must stay in [0, 2^31), matching the legacy
// `std::uniform_int_distribution<unsigned>(0, (1u<<31)-1)` used by the
// Mersenne-Twister path.
BOOST_AUTO_TEST_CASE(dither_range_and_determinism) {
  const std::uint64_t seed = 0x0123456789ABCDEFull;
  const std::uint32_t seedLo = static_cast<std::uint32_t>(seed);
  const std::uint32_t seedHi = static_cast<std::uint32_t>(seed >> 32);
  for (std::uint64_t block = 0; block != 8; ++block) {
    for (std::uint64_t symbol = 0; symbol != 64; ++symbol) {
      const std::uint32_t value =
          ditherValue(seedLo, seedHi, block, symbol);
      BOOST_CHECK_LT(value, 1u << 31);
      // Pure function: repeated calls give the same value.
      BOOST_CHECK_EQUAL(value, ditherValue(seedLo, seedHi, block, symbol));
    }
    // The first symbol of two adjacent blocks must differ (block is part of
    // the counter), otherwise dithering would repeat across time blocks.
    BOOST_CHECK_NE(ditherValue(seedLo, seedHi, block, 0),
                   ditherValue(seedLo, seedHi, block + 1, 0));
  }
}

BOOST_AUTO_TEST_SUITE_END()
