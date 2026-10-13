{
  description = "Dysco: compressing storage manager for Casacore MeasurementSets";

  inputs = {
    # Same nixpkgs release that astro-nix pins, for ABI-consistent
    # dependencies.  casacore and the ISPC compiler come from astro-nix.
    nixpkgs.url = "github:NixOS/nixpkgs/25.11";
    # Pinned to the revision used by rastro-io; later astro-nix commits are
    # not on GitHub yet and the GitHub HEAD has a broken default.nix.
    astro-nix.url = "github:adevress/astro-nix/bd2e7d9ad3246832782d8d2b35641b337ae3b227";
  };

  outputs =
    { self, nixpkgs, astro-nix }:
    let
      systems = [
        "aarch64-linux"
        "x86_64-linux"
      ];

      forAllSystems =
        f:
        nixpkgs.lib.genAttrs systems (
          system:
          let
            # astro-nix's legacyPackages already contains the whole nixpkgs
            # set plus the astronomy recipes (casacore) and `ispc`.
            astro-pkgs = astro-nix.legacyPackages.${system};
          in
          f { inherit system; pkgs = astro-pkgs; }
        );
    in
    {
      devShells = forAllSystems (
        { pkgs, ... }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              casacore
              gsl
              boost
              cmake
              pkg-config
              gnumake
              gcc
              clang
              ispc
            ];

            # The in-tree CMake/FindCasacore.cmake locates headers via
            # CASACORE_ROOT_DIR; cmake needs the prefix for its own searches.
            shellHook = ''
              export CASACORE_ROOT_DIR="${pkgs.casacore}"
              export CMAKE_PREFIX_PATH="${pkgs.casacore};${pkgs.gsl};${pkgs.boost};''${CMAKE_PREFIX_PATH:-}"
            '';
          };
        }
      );

      # Buildable package of this checkout with the ISPC backend enabled.
      # (CUDA is intentionally left to the caller: it needs the host driver
      # and the system CUDA toolkit.)
      packages = forAllSystems (
        { pkgs, ... }:
        {
          default = pkgs.stdenv.mkDerivation {
            pname = "dysco-ispc";
            version = "unstable";
            src = self;
            nativeBuildInputs = [
              pkgs.cmake
              pkgs.ispc
              pkgs.pkg-config
            ];
            buildInputs = [
              pkgs.casacore
              pkgs.gsl
              pkgs.boost
            ];
            cmakeFlags = [
              "-DENABLE_ISPC=ON"
              "-DPORTABLE=ON"
              "-DBUILD_PACKAGES=OFF"
            ];
            enableParallelBuilding = true;
          };
        }
      );
    };
}
