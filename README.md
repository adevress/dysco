The Dysco compression technique is explained in the article "Compression of interferometric radio-astronomical data",
A. R. Offringa (2016; http://arxiv.org/abs/1609.02019). If you use this software, please cite the paper.

# dysco
A compressing storage manager for Casacore mearement sets

To install:

    mkdir build
    cd build
    cmake ../
    make -j 4
    make install

To be able to open compressed measurement sets, the dysco library ("libdyscostman.so") must be in your path.

The Dysco compression technique is explained in [the article](http://arxiv.org/abs/1609.02019). Further documentation for the storage manager, including documentation of the `dscompress` tool:

https://github.com/aroffringa/dysco/wiki

# Accelerated backends

Dysco has three interchangeable implementations of the AF normalization /
quantization pipeline. All of them produce the same on-disk format, so any
backend can decode a file produced by another one:

| backend | build flag           | `dscompress` / `decompress` flag | notes |
|---------|----------------------|----------------------------------|-------|
| CPU     | (always built)       | (default)                        | Philox counter-based dither, greedy fit-to-max |
| CUDA    | `-DENABLE_CUDA=ON`   | `-cuda`                          | one kernel per stage in `kernels/cuda/` |
| ISPC    | `-DENABLE_ISPC=ON`   | `-ispc`                          | one ISPC kernel per stage in `kernels/ispc/` |

Both accelerated backends use the counter-based Philox dither
(`kernels/philox_rng.h`), so the symbol stream is independent of the thread /
SIMD layout. The CUDA and ISPC backends also share the parallel
damped-simultaneous ("Jacobi") fit-to-maximum algorithm.

## Building the ISPC backend

The ISPC compiler is provided by [astro-nix](https://github.com/adevress/astro-nix).
Enter its dev shell and configure with `ENABLE_ISPC`:

    nix develop
    cmake -S . -B build -DENABLE_ISPC=ON
    cmake --build build -j

`CMake` picks an ISPC target from the host architecture (`neon-i32x4` on
aarch64, `avx2-i32x8` on x86-64); override it with
`-DISPC_TARGET=avx512skx-x16`.


