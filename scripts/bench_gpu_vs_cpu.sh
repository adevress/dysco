#!/usr/bin/env bash
#
# End-to-end GPU vs CPU benchmark and bit-compatibility check.
#
# Usage: scripts/bench_gpu_vs_cpu.sh <original.ms> <workdir>
#
# The script compresses copies of <original.ms> with the CPU and GPU backends,
# times each run, and prints the sha256 of the compressed DATA column.  Runs
# that share the same backend settings must have identical checksums.
set -euo pipefail

MS="${1:?usage: bench_gpu_vs_cpu.sh <original.ms> <workdir>}"
WORK="${2:-$(mktemp -d)}"
BUILD="${BUILD:-build-cuda}"
DSCOMPRESS="$BUILD/dscompress"
DECOMPRESS="$BUILD/decompress"

# The nix-linked binaries cannot find the NVIDIA driver through the nix
# ld.so.cache; preload the real one.
if [ -e /usr/lib/aarch64-linux-gnu/libcuda.so.1 ]; then
  export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/aarch64-linux-gnu/libcuda.so.1}"
fi

mkdir -p "$WORK"

run_case() {
  local name="$1"; shift
  local dst="$WORK/$name.ms"
  rm -rf "$dst"
  cp -r "$MS" "$dst"
  chmod -R u+w "$dst"
  local t0 t1
  t0=$(date +%s.%N)
  "$DSCOMPRESS" "$@" "$dst" > "$WORK/$name.log" 2>&1
  t1=$(date +%s.%N)
  local wall comp
  wall=$(echo "$t1 - $t0" | bc)
  comp=$(grep -o 'Compression time: [0-9:.]*' "$WORK/$name.log" | awk '{print $3}')
  local hash
  hash=$(sha256sum "$dst/table.f1" | cut -d' ' -f1)
  printf "%-16s %10s %10s %s\n" "$name" "$comp" "$wall" "$hash"
}

printf "%-16s %10s %10s %s\n" "run" "compress" "wall[s]" "sha256(table.f1)"
run_case mt        -rng mt                    -static-seed
run_case phi       -rng philox                -static-seed
run_case phi_nofit -rng philox -no-fit-to-max -static-seed
run_case gpu_fit   -cuda -rng philox          -static-seed
run_case gpu_nofit -cuda -rng philox -no-fit-to-max -static-seed

# Decode one compressed file on both backends and compare.
rm -rf "$WORK/dec_cpu.ms" "$WORK/dec_gpu.ms"
cp -r "$WORK/phi_nofit.ms" "$WORK/dec_cpu.ms"
cp -r "$WORK/phi_nofit.ms" "$WORK/dec_gpu.ms"
"$DECOMPRESS"             "$WORK/dec_cpu.ms" > /dev/null
"$DECOMPRESS" -cuda        "$WORK/dec_gpu.ms" > /dev/null
echo
echo "decoded DATA checksums (CPU vs GPU decode):"
echo "  cpu: $(sha256sum "$WORK/dec_cpu.ms/table.f0" | cut -d' ' -f1)"
echo "  gpu: $(sha256sum "$WORK/dec_gpu.ms/table.f0" | cut -d' ' -f1)"
