{ pkgs ? import (import ./nix/sources.nix).nixpkgs {} }:

let
  sources = import ./nix/sources.nix;
  crateTools = pkgs.callPackage "${sources.crate2nix}/tools.nix" {};
  cargoNix = pkgs.callPackage (crateTools.generatedCargoNix {
    name = "finalfrontier";
    src = pkgs.nix-gitignore.gitignoreSource [ ".git/" "nix/" "*.nix" ] ./.;
  }) { inherit buildRustCrate; };
  crateOverrides = with pkgs; defaultCrateOverrides // {
    finalfrontier = attr: rec {
      pname = "finalfrontier";
      name = "${pname}-${attr.version}";

      nativeBuildInputs = [ gnumake installShellFiles pandoc ];

      buildInputs = stdenv.lib.optionals stdenv.isDarwin [
        darwin.Security
        libiconv
        openssl
      ];

      postBuild = ''
        # Build man pages.
        ( cd man ; make )

        # Generate shell completion files.
        for shell in bash fish zsh; do
          target/bin/finalfrontier completions $shell > finalfrontier.$shell
        done
      '';

      postInstall = ''
        # Install man pages.
        mkdir -p "$out/share/man/man1"
        cp man/*.1 "$out/share/man/man1/"

        # Install shell completions
        installShellCompletion finalfrontier.{bash,fish,zsh}
      '';

      meta = with stdenv.lib; {
        description = "Train word and subword embeddings";
        license = licenses.asl20;
        maintainers = with maintainers; [ danieldk ];
        platforms = platforms.all;
      };
    };
  };
  buildRustCrate = pkgs.buildRustCrate.override {
    defaultCrateOverrides = crateOverrides;
  };
in cargoNix.rootCrate.build
