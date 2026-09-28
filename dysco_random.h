#ifndef DYSCO_RANDOM_H
#define DYSCO_RANDOM_H

namespace dyscostman {

/**
 * Random generator used for quantization dithering.
 *
 * kMersenneTwister is the historical, sequential generator
 * (`std::mt19937`).  Its output depends on thread scheduling, so it is not
 * reproducible across CPU/GPU or even between runs with multiple threads.
 *
 * kPhilox is the counter-based Random123 generator.  A dither value is a pure
 * function of (seed, block index, symbol index), which makes the CPU and GPU
 * encoders bit-compatible.
 */
enum class RandomGenerator { kMersenneTwister, kPhilox };

}  // namespace dyscostman

#endif  // DYSCO_RANDOM_H
