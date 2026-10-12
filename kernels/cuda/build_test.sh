#!/usr/bin/env bash
# Quick standalone build of the GPU codec bit-compatibility test.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
"$NVCC" -O3 -std=c++17 -DDYSCO_USE_CUDA -I"$root" -I"$root/kernels" \
  "$root"/kernels/cuda/k*.cu "$root"/kernels/cuda/d*.cu \
  "$root"/kernels/cuda/gpu_af_codec.cu "$root"/kernels/af_reference.cc \
  "$root"/kernels/cuda/test_af_codec.cu -o /tmp/test_af_codec
/tmp/test_af_codec
