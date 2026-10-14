#include "../kernels/rng.h"

#include <boost/test/unit_test.hpp>

#include <cstdint>

// Tests for the dither RNG facade (kernels/rng.h), which uses alea's threefry
// generator.  They check the properties the reproducibility of the encoders
// relies on: a fixed domain range, determinism, and sensitivity to both the
// block index and the phase.

BOOST_AUTO_TEST_SUITE(rng_facade)

namespace {

constexpr std::uint64_t kSeed = 0x0123456789ABCDEFull;
constexpr std::uint32_t kSeedLo = static_cast<std::uint32_t>(kSeed);
constexpr std::uint32_t kSeedHi = static_cast<std::uint32_t>(kSeed >> 32);

}  // namespace

// The dither must stay in [0, 2^31), the range the quantizer's threshold is
// computed for.
BOOST_AUTO_TEST_CASE(dither_range) {
  for (std::uint64_t block = 0; block != 8; ++block)
    for (std::uint64_t symbol = 0; symbol != 64; ++symbol)
      BOOST_CHECK_LT(
          dyscostman::rng::ditherValue(kSeedLo, kSeedHi, block, symbol),
          1u << 31);
}

// A pure function of (seed, block, symbol): repeated calls are identical.
BOOST_AUTO_TEST_CASE(determinism) {
  for (std::uint64_t block = 0; block != 4; ++block)
    for (std::uint64_t symbol = 0; symbol != 32; ++symbol)
      BOOST_CHECK_EQUAL(
          dyscostman::rng::ditherValue(kSeedLo, kSeedHi, block, symbol),
          dyscostman::rng::ditherValue(kSeedLo, kSeedHi, block, symbol));
}

// The block index is part of the counter: adjacent time blocks must not share
// the same dither sequence.
BOOST_AUTO_TEST_CASE(block_sensitivity) {
  for (std::uint64_t block = 0; block != 8; ++block)
    BOOST_CHECK_NE(
        dyscostman::rng::ditherValue(kSeedLo, kSeedHi, block, 0),
        dyscostman::rng::ditherValue(kSeedLo, kSeedHi, block + 1, 0));
}

// Real and imaginary parts of the same visibility use disjoint symbols.
BOOST_AUTO_TEST_CASE(phase_separation) {
  for (std::uint64_t visibility = 0; visibility != 32; ++visibility)
    BOOST_CHECK_NE(dyscostman::rng::ditherForVisibility(kSeedLo, kSeedHi, 0,
                                                        visibility, 0),
                   dyscostman::rng::ditherForVisibility(kSeedLo, kSeedHi, 0,
                                                        visibility, 1));
}

// The compiled-in backend is named.
BOOST_AUTO_TEST_CASE(backend_name) {
  BOOST_CHECK(dyscostman::rng::kName != nullptr);
  BOOST_TEST_MESSAGE("dither backend: " << dyscostman::rng::kName);
}

BOOST_AUTO_TEST_SUITE_END()
