// K4 (ISPC) — iterative antenna-gain solver.
//
// ISPC port of `kernels/cuda/k4_antenna_gain_solve.cu`.  Solves
// |V_ij| ~= g_i * g_j in a weighted, damped Jacobi fashion.  The outer loop is
// sequential (it mirrors the CPU reference exactly), while the per-antenna
// work inside each iteration is vectorized across lanes: each lane owns one
// antenna `a` and walks `j` in the fixed sequential order, so the floating
// point accumulation order — and therefore the result — is unchanged.
//
// The two block-wide maxima are order-independent, so they are computed with
// plain scalar loops after the vectorized updates.  This avoids relying on
// foreach reductions while keeping the hot inner `j` loop SIMD.

#include "ispc_common.isph"

export void ispc_k4_antenna_gain_solve(const uniform double matrix[],
                                       uniform double rms[],
                                       uniform double next[],
                                       uniform int nAnt) {
  const int kMaxIterations = 100;
  const double kPrecisionTarget = 1e-6;
  const double kDampingNew = 0.8;
  const double kDampingOld = 0.2;
  const double kZeroThreshold = 1e-5;

  foreach (a = 0 ... nAnt) rms[a] = 1.0;

  uniform double precision = 1.0;
  uniform int iteration = 0;
  while (iteration < kMaxIterations && precision > kPrecisionTarget) {
    ++iteration;

    // next[a] = sum_j matrix(a,j) / sum_j rms[j] over finite off-diagonal
    // entries with a non-zero gain, accumulated over j in order.
    foreach (a = 0 ... nAnt) {
      double weightSum = 0.0;
      double acc = 0.0;
      for (int j = 0; j < nAnt; ++j) {
        if (a != j && rms[j] != 0.0) {
          const double m = matrix[a * nAnt + j];
          if (isfinite(m)) {
            acc += m;
            weightSum += rms[j];
          }
        }
      }
      next[a] = (weightSum == 0.0) ? 0.0 : acc / weightSum;
    }

    // Damped update, then the block-wide maximum over the new gains.
    foreach (a = 0 ... nAnt) {
      rms[a] = next[a] * kDampingNew + rms[a] * kDampingOld;
    }
    uniform double maxVal = 0.0;
    for (uniform int a = 0; a < nAnt; ++a) maxVal = max(maxVal, rms[a]);

    // Zero the gains below the relative threshold and measure convergence.
    uniform double newPrecision = 0.0;
    for (uniform int a = 0; a < nAnt; ++a) {
      if (rms[a] < maxVal * kZeroThreshold) rms[a] = 0.0;
      const uniform double relativeChange = abs(rms[a] - next[a]) / maxVal;
      newPrecision = max(newPrecision, relativeChange);
    }
    precision = newPrecision;
  }
}
