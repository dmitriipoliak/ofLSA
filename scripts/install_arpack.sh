#!/usr/bin/env bash
# Build arpack-ng with MPI (PARPACK) into a user directory. No root needed.
#
# Requirements: cmake >= 3.10, an MPI Fortran compiler (mpif90) and BLAS/LAPACK.
# On a cluster this is typically:   module load gcc openmpi cmake openblas
#
# Usage:  scripts/install_arpack.sh [prefix]
#         default prefix: $HOME/software/arpack-ng-install

set -euo pipefail

PREFIX="${1:-$HOME/software/arpack-ng-install}"
VERSION="${ARPACK_NG_VERSION:-3.9.1}"
BUILD_ROOT="${TMPDIR:-/tmp}/arpack-ng-build-$$"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

for tool in cmake mpif90 mpicc; do
    command -v "$tool" >/dev/null || { echo "Missing: $tool (load a module or install it)"; exit 1; }
done

mkdir -p "$BUILD_ROOT"
cd "$BUILD_ROOT"

echo "Downloading arpack-ng $VERSION ..."
if command -v curl >/dev/null; then
    curl -fsSL -o arpack-ng.tar.gz "https://github.com/opencollab/arpack-ng/archive/refs/tags/${VERSION}.tar.gz"
else
    wget -q -O arpack-ng.tar.gz "https://github.com/opencollab/arpack-ng/archive/refs/tags/${VERSION}.tar.gz"
fi
tar xzf arpack-ng.tar.gz
cd "arpack-ng-${VERSION}"

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI=ON \
    -DICB=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DEXAMPLES=OFF \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_C_COMPILER=mpicc

cmake --build build -j "$JOBS"
cmake --install build

cd /
rm -rf "$BUILD_ROOT"

echo
echo "PARPACK installed to: $PREFIX"
echo "etc/bashrc will pick it up automatically (or: export ARPACK_DIR=$PREFIX)"
