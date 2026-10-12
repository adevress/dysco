{
  description = "Dysco: compressing storage manager for Casacore MeasurementSets";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixpkgs-unstable/nixexprs.tar.zst";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-linux" "x86_64-linux" ];
      forAllSystems = f:
        nixpkgs.lib.genAttrs systems
          (system: f (import nixpkgs { inherit system; }));
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            casacore
            gsl
            boost
            cmake
            pkg-config
            gnumake
            gcc
          ];

          # The in-tree CMake/FindCasacore.cmake locates headers via
          # CASACORE_ROOT_DIR; cmake needs the prefix for its own searches.
          shellHook = ''
            export CASACORE_ROOT_DIR="${pkgs.casacore}"
            export CMAKE_PREFIX_PATH="${pkgs.casacore};${pkgs.gsl};${pkgs.boost};''${CMAKE_PREFIX_PATH:-}"
          '';
        };
      });
    };
}
