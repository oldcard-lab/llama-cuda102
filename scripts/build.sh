#!/bin/bash
# Fetch llama.cpp v1.0.0, apply the patch series, and build the CUDA backend for Kepler (sm_30) with CUDA 10.2,
# using clang as the CUDA compiler (CUDA 10.2's nvcc only accepts old host compilers).
#
# Usage: scripts/build.sh [WORK_DIR] [CUDA_DIR]
#   WORK_DIR  where llama.cpp is cloned and built (default: ./work)
#   CUDA_DIR  CUDA 10.2 toolkit (default: $HOME/cuda-10.2, see scripts/install-cuda-10.2.sh)
# Env: CLANG_VERSION (default 22: clang-22, clang++-22, ld.lld-22 must be on PATH), JOBS (default: nproc),
#      TARGETS (default: llama-cli llama-server llama-completion llama-bench test-backend-ops).
# Result: WORK_DIR/llama.cpp/build/bin/*
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mkdir -p "${1:-work}" && cd "${1:-work}" && pwd)
CUDA_DIR=${2:-$HOME/cuda-10.2}
V=${CLANG_VERSION:-22}
JOBS=${JOBS:-$(nproc)}
TARGETS=${TARGETS:-llama-cli llama-server llama-completion llama-bench test-backend-ops}
UPSTREAM=https://github.com/ggml-org/llama.cpp.git
BASE_TAG=v1.0.0
BASE_COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28

for t in clang-$V clang++-$V ld.lld-$V cmake ninja git; do
  command -v "$t" > /dev/null || { echo "missing tool: $t"; exit 2; }
done
[ -f "$CUDA_DIR/version.txt" ] && grep -q 'CUDA Version 10.2' "$CUDA_DIR/version.txt" ||
  { echo "CUDA 10.2 toolkit not found in $CUDA_DIR (scripts/install-cuda-10.2.sh)"; exit 2; }

SRC=$WORK/llama.cpp
if [ ! -d "$SRC/.git" ]; then
  git clone --branch "$BASE_TAG" --depth 1 "$UPSTREAM" "$SRC"
fi
if ! git -C "$SRC" merge-base --is-ancestor "$BASE_COMMIT" HEAD 2>/dev/null; then
  echo "$SRC is not llama.cpp $BASE_TAG ($BASE_COMMIT); it may be from an older version of this repository."
  echo "Delete $SRC and run again."
  exit 3
fi
subjects=$(git -C "$SRC" log --format=%s)
# fingerprint of patches/, written when they are applied: a rerun with the same files skips git am, a work folder
# that carries another version of the series (same commit subjects, different code) is refused
PSUM=$(cat "$HERE"/patches/*.patch | sha256sum | cut -d' ' -f1)
MARK="$SRC/.git/llama-cuda102-series"
if echo "$subjects" | grep -q '6uO, llama.cpp discussion #29961' &&
   echo "$subjects" | grep -q 'FunctionGemma tool-call parser'; then
  if [ "$(cat "$MARK" 2>/dev/null)" != "$PSUM" ]; then
    echo "$SRC carries a different version of this patch series. Delete $SRC and run again."
    exit 4
  fi
elif [ "$(git -C "$SRC" rev-parse HEAD)" = "$BASE_COMMIT" ]; then
  # --keep-cr: one file in patch 0003 has CRLF line endings, which plain `git am` would strip
  git -C "$SRC" -c user.name=oldcard-lab -c user.email=oldcard-lab@localhost am --keep-cr "$HERE"/patches/*.patch
  echo "$PSUM" > "$MARK"
else
  echo "$SRC has other commits on top of $BASE_TAG (an older or partly applied series). Delete it and run again."
  exit 4
fi

cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang-$V -DCMAKE_CXX_COMPILER=clang++-$V \
  -DCMAKE_CUDA_COMPILER=clang++-$V -DCMAKE_CUDA_HOST_COMPILER=clang++-$V -DCUDAToolkit_ROOT="$CUDA_DIR" \
  -DCMAKE_CUDA_ARCHITECTURES=30 \
  -DCMAKE_CUDA_FLAGS="--cuda-path=$CUDA_DIR --no-cuda-version-check -Wno-unknown-cuda-version" \
  -DCMAKE_EXE_LINKER_FLAGS="--ld-path=$(command -v ld.lld-$V) -Wl,--disable-new-dtags" \
  -DCMAKE_SHARED_LINKER_FLAGS="--ld-path=$(command -v ld.lld-$V) -Wl,--disable-new-dtags" \
  -DCMAKE_BUILD_RPATH="$SRC/build/bin;$CUDA_DIR/lib64" \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF -DGGML_CUDA_NO_VMM=ON \
  -DLLAMA_BUILD_HTML=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
# (LLAMA_BUILD_HTML / LLAMA_USE_PREBUILT_UI off: llama-server's web UI assets otherwise need a Node.js toolchain at
# build time; the server's API works without them.)
# shellcheck disable=SC2086
cmake --build "$SRC/build" -j "$JOBS" --target $TARGETS
echo "built: $SRC/build/bin"
