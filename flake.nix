{
  description = "json-clankers — local AI model experimentation and agent patterns.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
      ];
      perSystem = { pkgs, ... }:
        let
          unfree-pkgs = import inputs.nixpkgs {
            system = pkgs.system;
            config.allowUnfreePredicate = pkg:
              builtins.elem (pkgs.lib.getName pkg) [ "open-webui" ];
          };
        in
        {
          formatter = pkgs.nixfmt;
          devShells.default = pkgs.mkShell {
            packages = [
              pkgs.ollama
              unfree-pkgs.open-webui
              pkgs.python3
              pkgs.uv
            ];
          };
        };
    };
}
