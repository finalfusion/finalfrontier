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

cp "target/${TARGET}/release/ff-train" "${staging}/ff-train"
strip "${staging}/ff-train"
cp {README.md,LICENSE,NOTICE} "${staging}/"
cp man/*.1 "${staging}/"

( cd "${tmpdir}" && tar czf "${out_dir}/${name}.tar.gz" "${name}")

rm -rf "${tmpdir}"
