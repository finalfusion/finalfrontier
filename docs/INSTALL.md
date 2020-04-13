# Installation

## Using Nix

finalfrontier can be directly installed from its repository using the
[Nix](https://nixos.org/nix/) package manager. To install the current
version from the `master` branch into your user profile:

```bash
$ nix-env -i \
  -f https://github.com/finalfusion/finalfrontier/tarball/nix-build
```

You can get prebuilt Linux/macOS binaries using the [finalfusion
Cachix cache](https://finalfusion.cachix.org):

```bash
# If you haven't installed Cachix yet:
$ nix-env -iA cachix -f https://cachix.org/api/v1/install
$ cachix use finalfusion
```

## From source

There are currently no pre-compiled finalfrontier binaries. Compilation
requires a working [Rust](https://www.rust-lang.org/) toolchain. After
installing Rust, you can compile finalfrontier using cargo:

~~~shell
$ cargo install finalfrontier-utils
~~~

Afterwards, the binaries are available in your `~/.cargo/bin`.
