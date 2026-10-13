// K2 (ISPC) — per (channel, polarization) RMS normalization factor.
//
// ISPC port of `kernels/cuda/k2_channel_rms.cu`.  One lane per
// (channel,pol) plane walks the rows in order and accumulates re^2 + im^2 in
// double.  The sequential row loop is what keeps the reduction bit-identical
// to the CPU encoder; the SIMD parallelism is across the (channel,pol) planes,
// which is exactly where the independent work is.

#include "ispc_common.isph"

export void ispc_k2_channel_rms(const uniform double re[],
                                const uniform double im[],
                                uniform float channelRms[],
                                uniform int nRows, uniform int visPerRow) {
  foreach (i = 0 ... visPerRow) {
    double value = 0.0;
    int count = 0;
    for (int row = 0; row < nRows; ++row) {
      const int idx = row * visPerRow + i;
      const double r = re[idx];
      const double v = im[idx];
      if (isfinite(r) && isfinite(v)) {
        ++count;
        value += r * r + v * v;
      }
    }
    channelRms[i] = (float)sqrt(value / ((double)count * 2.0));
  }
}
