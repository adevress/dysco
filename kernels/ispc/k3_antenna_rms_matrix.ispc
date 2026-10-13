// K3 (ISPC) — per-baseline RMS matrix used by the antenna-gain solver.
//
// ISPC port of `kernels/cuda/k3_antenna_rms_matrix.cu`.  One work-item per
// row computes the RMS over the channels of its own row (a contiguous SoA
// read), then scatters the two symmetric matrix entries.  The initial NaN
// fill marks absent baselines / autocorrelations as non-finite so the solver
// skips them, exactly like the CPU code.

#include "ispc_common.isph"

export void ispc_k3_antenna_rms_matrix(
    const uniform double re[], const uniform double im[],
    const uniform int ant1[], const uniform int ant2[], uniform double matrix[],
    uniform int nRows, uniform int nChannels, uniform int nPol,
    uniform int pol, uniform int nAnt, uniform double nanValue) {
  const uniform int visPerRow = nChannels * nPol;

  // NaN-fill: the matrix is nAnt*nAnt but only the triangle is written below.
  foreach (i = 0 ... nAnt * nAnt) matrix[i] = nanValue;

  foreach (row = 0 ... nRows) {
    int a1 = ant1[row];
    int a2 = ant2[row];
    if (a1 == a2) continue;  // autocorrelations have no baseline RMS
    if (a1 > a2) {
      const int t = a1;
      a1 = a2;
      a2 = t;
    }
    double value = 0.0;
    int count = 0;
    for (int ch = 0; ch < nChannels; ++ch) {
      const int idx = row * visPerRow + ch * nPol + pol;
      const double r = re[idx];
      const double v = im[idx];
      if (isfinite(r) && isfinite(v)) {
        ++count;
        value += r * r + v * v;
      }
    }
    const double rms = sqrt(value / ((double)count * 2.0));
    matrix[a1 * nAnt + a2] = rms;
    matrix[a2 * nAnt + a1] = rms;
  }
}
