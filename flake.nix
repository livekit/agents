{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, utils, nixpkgs }:
    utils.lib.eachDefaultSystem (system:
      let pkgs = (import nixpkgs) {
        inherit system;
      };

      in {
        devShell = with pkgs; mkShell {
          nativeBuildInputs = [ python3 virtualenv ruff ];
          shellHook = ''
            virtualenv venv
            source venv/bin/activate
          '';
        };
      }
    );
}
