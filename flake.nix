{
  description = "External provider plugins for contextualize";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
  };

  outputs = { nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        pyprojectOverrides = final: prev: {
          grapheme = prev.grapheme.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {
              setuptools = [];
            };
          });
          pylatexenc = prev.pylatexenc.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {
              setuptools = [];
            };
          });
        };
        pythonSet =
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope (
            nixpkgs.lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
              overlay
              pyprojectOverrides
            ]
          );
        venv = pythonSet.mkVirtualEnv "cx-plugins-env" workspace.deps.default;
      in
      {
        packages.default = venv;
        packages.cx-plugins = pythonSet."cx-plugins";

        checks.default = venv;

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.git
            pkgs.python312
            pkgs.uv
          ];
          env = {
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
        };
      });
}
