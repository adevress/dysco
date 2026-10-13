// K5 (ISPC) — apply the normalization factors.
//
// ISPC port of `kernels/cuda/k5_apply_norm.cu`.  K5a divides every value by
// its channel RMS, K5b multiplies by g_a1 * g_a2 for the polarization being
// processed.  Both are element-wise over contiguous SoA data: no gathers and
// no scatters.  The operation order matches the CPU encoder.

#include "ispc_common.isph"

export void ispc_k5a_apply_channel_norm(uniform double re[],
                                        uniform double im[],
                                        const uniform float channelRms[],
                                        uniform int nVis,
                                        uniform int visPerRow) {
  foreach (i = 0 ... nVis) {
    const double rms = (double)channelRms[i % visPerRow];
    // Match the CPU encoder: a zero RMS leaves the (zero) values untouched
    // instead of dividing 0/0 and producing NaNs.
    if (rms != 0.0) {
      re[i] /= rms;
      im[i] /= rms;
    }
  }
}

export void ispc_k5b_apply_antenna_norm(
    uniform double re[], uniform double im[], const uniform int ant1[],
    const uniform int ant2[], const uniform double rmsPerAntenna[],
    uniform int nRows, uniform int nChannels, uniform int nPol,
    uniform int pol) {
  const uniform int visPerRow = nChannels * nPol;
  foreach (row = 0 ... nRows) {
    const double mul = rmsPerAntenna[ant1[row]] * rmsPerAntenna[ant2[row]];
    const double fac = (mul == 0.0) ? 0.0 : 1.0 / mul;
    for (int ch = 0; ch < nChannels; ++ch) {
      const int idx = row * visPerRow + ch * nPol + pol;
      re[idx] *= fac;
      im[idx] *= fac;
    }
  }
}
