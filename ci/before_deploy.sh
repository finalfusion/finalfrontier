#!/bin/bash

set -ex

cargo build --target "$TARGET" --release

( cd man ; make )

tmpdir="$(mktemp -d)"
name="${PROJECT_NAME}-${TRAVIS_TAG}-${TARGET}"
staging="${tmpdir}/${name}"
mkdir "${staging}"
out_dir="$(pwd)/deployment"
mkdir "${out_dir}"

cp "target/${TARGET}/release/finalfrontier" "${staging}/finalfrontier"
strip "${staging}/finalfrontier"
cp {README.md,LICENSE,NOTICE} "${staging}/"
cp man/*.1 "${staging}/"

( cd "${tmpdir}" && tar czf "${out_dir}/${name}.tar.gz" "${name}")

rm -rf "${tmpdir}"
