// K6 (ISPC) — "fit to maximum" post-normalization for the AF encoder.
//
// ISPC port of the default, grid-parallel *damped simultaneous* ("Jacobi")
// algorithm in `kernels/cuda/k6_fit_to_max.cu`.  As on the GPU, the algorithm
// only stores the resulting per-channel / per-antenna metadata factors, so the
// decoder is agnostic to how they were produced.
//
// All the per-element passes are contiguous and vectorized; the (completely
// scalar) per-channel / per-antenna maxima are kept in the same order as the
// CUDA and CPU versions, so the metadata is deterministic and matches the
// other backends to within the tiny `cbrt` rounding differences.
//
// The exported functions mirror the CUDA kernels one for one:
//   ispc_k6a_reduce_channel_max  <-> k6a_reduce_channel_max
//   ispc_k6a_apply_channel_fit   <-> k6a_apply_channel_fit
//   ispc_k6b_reduce              <-> k6b_reduce
//   ispc_k6b_factors             <-> k6b_factors
//   ispc_k6b_apply               <-> k6b_apply

#include "ispc_common.isph"

// Phase A: per-visibility (channel x pol) maximum over cross-correlation rows.
export void ispc_k6a_reduce_channel_max(
    const uniform double re[], const uniform double im[],
    const uniform int ant1[], const uniform int ant2[], uniform double L[],
    uniform int nRows, uniform int visPerRow) {
  foreach (vi = 0 ... visPerRow) {
    double best = 0.0;
    for (int row = 0; row < nRows; ++row) {
      if (ant1[row] != ant2[row]) {
        const int idx = row * visPerRow + vi;
        const double c = ispc_component_max(re[idx], im[idx]);
        if (isfinite(c) && c > best) best = c;
      }
    }
    L[vi] = best;
  }
}

// Phase A: factor maxLevel/L[vi] (no max(1,.) clamp: may downscale), divide the
// channel metadata by it, and scale every element including autocorrelations.
export void ispc_k6a_apply_channel_fit(
    uniform double re[], uniform double im[], const uniform double L[],
    uniform float meta[], uniform int nRows, uniform int visPerRow,
    uniform double maxLevel) {
  foreach (vi = 0 ... visPerRow) {
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    meta[vi] = (float)((double)meta[vi] / g0);
  }
  const uniform int n = nRows * visPerRow;
  foreach (i = 0 ... n) {
    const int vi = i % visPerRow;
    const double l = L[vi];
    const double g0 = (maxLevel == 0.0 || l == 0.0) ? 1.0 : maxLevel / l;
    re[i] *= g0;
    im[i] *= g0;
  }
}

// Phase B: per-round channel maxima L[vi] and antenna maxima M[pol*nAnt+ant]
// over cross-correlation rows.
export void ispc_k6b_reduce(const uniform double re[], const uniform double im[],
                            const uniform int ant1[], const uniform int ant2[],
                            uniform double L[], uniform double M[],
                            uniform int nRows, uniform int visPerRow,
                            uniform int nChannels, uniform int nPol,
                            uniform int nAnt) {
  foreach (vi = 0 ... visPerRow) {
    double best = 0.0;
    for (int row = 0; row < nRows; ++row) {
      if (ant1[row] == ant2[row]) continue;
      const int idx = row * visPerRow + vi;
      const double c = ispc_component_max(re[idx], im[idx]);
      if (isfinite(c) && c > best) best = c;
    }
    L[vi] = best;
  }

  const uniform int nM = nPol * nAnt;
  foreach (ai = 0 ... nM) {
    const int pol = ai / nAnt;
    const int ant = ai - pol * nAnt;
    double best = 0.0;
    for (int row = 0; row < nRows; ++row) {
      if (ant1[row] == ant2[row]) continue;
      if (ant1[row] != ant && ant2[row] != ant) continue;
      for (int ch = 0; ch < nChannels; ++ch) {
        const int idx = row * visPerRow + ch * nPol + pol;
        const double c = ispc_component_max(re[idx], im[idx]);
        if (isfinite(c) && c > best) best = c;
      }
    }
    M[ai] = best;
  }
}

// Phase B: turn the maxima into damped cube-root factors, divide the metadata
// by them and report whether any axis still has meaningful headroom.
export uniform int ispc_k6b_factors(uniform double L[], uniform double M[],
                                    uniform float meta[], uniform int nChPol,
                                    uniform int nPolAnt, uniform int visPerRow,
                                    uniform double maxLevel) {
  const uniform double kEpsChannel = 1e-3;
  const uniform double kEpsAntenna = 1e-2;
  uniform int progress = 0;
  for (uniform int vi = 0; vi < nChPol; ++vi) {
    const uniform double l = L[vi];
    const uniform double full =
        (maxLevel != 0.0 && l > 0.0) ? max((double)1.0, maxLevel / l) : 1.0;
    if (full > 1.0 + kEpsChannel) progress = 1;
    const uniform double g = cbrt(full);
    L[vi] = g;  // now the factor, not the maximum
    meta[vi] = (float)((double)meta[vi] / g);
  }
  for (uniform int ai = 0; ai < nPolAnt; ++ai) {
    const uniform double m = M[ai];
    const uniform double full =
        (maxLevel != 0.0 && m > 0.0) ? max((double)1.0, maxLevel / m) : 1.0;
    if (full > 1.0 + kEpsAntenna) progress = 1;
    const uniform double f = cbrt(full);
    M[ai] = f;
    meta[visPerRow + ai] = (float)((double)meta[visPerRow + ai] / f);
  }
  return progress;
}

// Phase B: elementwise application of G * F_{a1} * F_{a2} to all elements.
export void ispc_k6b_apply(uniform double re[], uniform double im[],
                           const uniform int ant1[], const uniform int ant2[],
                           const uniform double G[], const uniform double F[],
                           uniform int nRows, uniform int visPerRow,
                           uniform int nPol, uniform int nAnt) {
  const uniform int n = nRows * visPerRow;
  foreach (i = 0 ... n) {
    const int row = i / visPerRow;
    const int vi = i - row * visPerRow;
    const int pol = vi % nPol;
    // Autocorrelations (a1 == a2) get the antenna factor twice, matching the
    // CPU reference.
    const double scale =
        G[vi] * F[pol * nAnt + ant1[row]] * F[pol * nAnt + ant2[row]];
    re[i] *= scale;
    im[i] *= scale;
  }
}
