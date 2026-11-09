{
  description = "json-inference — local model serving and evaluation.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
      ];
      perSystem =
        { pkgs, ... }:
        {
          formatter = pkgs.nixfmt;
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];
            # `inference` runs straight from the checkout: it reads models/ and
            # writes results/ next to itself.
            shellHook = ''
              export PATH="$PWD/bin:$PATH"
            '';
          };
        };
    };
}
