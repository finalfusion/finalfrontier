{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  finalfrontier = import ./default.nix {
    inherit pkgs;
  };
in finalfrontier.override {
  runTests = true;
}
