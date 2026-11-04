#!/usr/bin/env bash
# Build arpack-ng with MPI (PARPACK) into a user directory. No root needed.
#
# Needs: an MPI Fortran/C compiler (mpif90, mpicc) — on a cluster: module load gcc openmpi
# Optional (built/installed automatically if missing):
#   cmake         -> installed with pip --user
#   BLAS/LAPACK   -> OpenBLAS built into the same prefix
#
# Usage:  scripts/install_arpack.sh [prefix]
#         default prefix: $HOME/software/arpack-ng-install
# Env:    ARPACK_NG_VERSION (default 3.9.1), OPENBLAS_VERSION (default 0.3.28)

set -euo pipefail

PREFIX="${1:-$HOME/software/arpack-ng-install}"
ARPACK_VERSION="${ARPACK_NG_VERSION:-3.9.1}"
OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.28}"
BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/oflsa-arpack-XXXXXX")"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

trap 'rm -rf "$BUILD_ROOT"' EXIT

download()
{
    if command -v curl >/dev/null; then curl -fsSL -o "$2" "$1"
    else wget -q -O "$2" "$1"; fi
}

# --- compilers -------------------------------------------------------------
for tool in mpif90 mpicc; do
    command -v "$tool" >/dev/null || {
        echo "Missing $tool. Load an MPI module (e.g. 'module load gcc openmpi')"
        echo "or install OpenMPI development packages."
        exit 1
    }
done

# --- cmake -----------------------------------------------------------------
if ! command -v cmake >/dev/null; then
    echo "cmake not found: installing it with pip --user"
    python3 -m pip install --user --quiet cmake
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- BLAS/LAPACK -----------------------------------------------------------
have_lapack()
{
    ldconfig -p 2>/dev/null | grep -qE "(liblapack|libopenblas|libmkl_rt)\.so " && return 0
    for d in ${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH//:/ }} "$PREFIX/lib" "$PREFIX/lib64"; do
        ls "$d"/liblapack.so* "$d"/libopenblas.so* >/dev/null 2>&1 && return 0
    done
    return 1
}

CMAKE_EXTRA=()
if ls "$PREFIX"/lib/libopenblas.so >/dev/null 2>&1; then
    echo "Using OpenBLAS already built in $PREFIX"
    CMAKE_EXTRA+=(
        "-DBLAS_LIBRARIES=$PREFIX/lib/libopenblas.so"
        "-DLAPACK_LIBRARIES=$PREFIX/lib/libopenblas.so"
    )
elif have_lapack; then
    echo "BLAS/LAPACK found on the system"
else
    echo "BLAS/LAPACK not found: building OpenBLAS $OPENBLAS_VERSION into $PREFIX"
    cd "$BUILD_ROOT"
    download "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" openblas.tar.gz
    tar xzf openblas.tar.gz
    cd "OpenBLAS-${OPENBLAS_VERSION}"
    make -j "$JOBS" USE_OPENMP=0 NUM_THREADS=64 DYNAMIC_ARCH=1 FC=gfortran >/dev/null
    make PREFIX="$PREFIX" install >/dev/null
    CMAKE_EXTRA+=(
        "-DBLAS_LIBRARIES=$PREFIX/lib/libopenblas.so"
        "-DLAPACK_LIBRARIES=$PREFIX/lib/libopenblas.so"
    )
fi

# --- arpack-ng ---------------------------------------------------------------
echo "Building arpack-ng $ARPACK_VERSION with MPI ..."
cd "$BUILD_ROOT"
if command -v git >/dev/null && \
   git clone -q --depth 1 --branch "$ARPACK_VERSION" https://github.com/opencollab/arpack-ng.git "arpack-ng-${ARPACK_VERSION}" 2>/dev/null
then
    :
else
    download "https://github.com/opencollab/arpack-ng/archive/refs/tags/${ARPACK_VERSION}.tar.gz" arpack-ng.tar.gz
    tar xzf arpack-ng.tar.gz
fi
cd "arpack-ng-${ARPACK_VERSION}"

cmake -S . -B build \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMPI=ON \
    -DICB=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DEXAMPLES=OFF \
    -DCMAKE_Fortran_COMPILER=mpif90 \
    -DCMAKE_C_COMPILER=mpicc \
    "${CMAKE_EXTRA[@]}" >/dev/null

cmake --build build -j "$JOBS" >/dev/null
cmake --install build >/dev/null

ls "$PREFIX"/lib*/libparpack.* >/dev/null 2>&1 || { echo "PARPACK build failed"; exit 1; }

echo
echo "PARPACK installed to: $PREFIX"
echo "Now: source etc/bashrc   (it finds $PREFIX automatically)"
