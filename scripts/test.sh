#!/bin/bash
# Correctness and speed check of a build from scripts/build.sh, on the Kepler GPU.
#   1. test-backend-ops, CUDA0 vs CPU: MUL_MAT and MUL_MAT_ID (the ops the Kepler kernels change), plus the
#      convolution ops (IM2COL, CONV_2D); with FULL=1 every op (takes several minutes).
#   2. tools/mmvk-check: strict per-row checks of the Kepler matrix-vector kernels against an fp64 reference
#      (NaN-poisoned output, bitwise repeat, every row within 2e-5), plain and weights-buffer mode.
#   3. Optional: if MODEL=<path to a .gguf> is set, greedy text on the GPU vs the CPU, and llama-bench speed.
# Usage: scripts/test.sh [WORK_DIR] [CUDA_DIR]      (same arguments as build.sh)
# Exit status 0 only if every check passed.
set -u
WORK=${1:-work}; CUDA_DIR=${2:-$HOME/cuda-10.2}; V=${CLANG_VERSION:-22}
HERE=$(cd "$(dirname "$0")/.." && pwd)
SRC=$(cd "$WORK/llama.cpp" && pwd); B=$SRC/build/bin
strip() { sed -r 's/\x1b\[[0-9;]*m//g'; }
fail=0

echo "== 1. test-backend-ops (CUDA0 vs CPU)"
ops="MUL_MAT MUL_MAT_ID IM2COL CONV_2D"; [ -n "${FULL:-}" ] && ops=ALL
for op in $ops; do
  a=(-b CUDA0); [ "$op" != ALL ] && a+=(-o "$op")
  out=$("$B/test-backend-ops" "${a[@]}" 2>&1 | strip)
  ok=$(echo "$out" | grep -cE ': OK$'); bad=$(echo "$out" | grep -E ': FAIL$' | grep -cv 'failed to allocate')
  big=$(echo "$out" | grep -ciE 'failed to allocate|out of memory')
  echo "$op: OK=$ok FAIL=$bad (cases too big for this card's memory: $big)"
  [ "$bad" -eq 0 ] || { fail=1; echo "$out" | grep -E ': FAIL$' | head -5; }
  echo "$out" | grep -qE 'GGML_ASSERT|Segmentation|Aborted' && { fail=1; echo "  $op run aborted"; }
done
echo "   (a tiny q4_1 / q5_1 MUL_MAT case occasionally fails by a hair with random inputs on any build; rerun it)"

echo "== 2. mmvk-check (strict Kepler kernel check)"
clang++-$V -O2 -std=c++17 "$HERE/tools/mmvk-check.cpp" -I"$SRC/ggml/include" -L"$B" \
  -lggml -lggml-base -lggml-cuda -Wl,-rpath,"$B" -Wl,-rpath,"$CUDA_DIR/lib64" \
  -Wl,-rpath-link,"$CUDA_DIR/lib64" -Wl,--disable-new-dtags -o "$B/mmvk-check" || { echo "mmvk-check build failed"; exit 1; }
NCT=$(grep -oE 'MMVK_NCOLS_TYPES: [A-Za-z0-9_,]+' "$SRC/ggml/src/ggml-cuda/mmvk.cu" | head -1 | cut -d' ' -f2)
for mode in "" --weights; do
  # every weight type the Kepler kernels handle (the release gate checks the same set: 597 cases per mode)
  out=$(MMVK_CHECK_NCOLS="$NCT" MMVK_CHECK_TYPES=q4_K,q6_K,q8_0,q5_0,f16,q3_K,iq3_xxs,q5_K "$B/mmvk-check" $mode 2>&1); rc=$?
  echo "$out" | grep -E '^SUMMARY' || echo "  no SUMMARY line (crash?)"
  echo "$out" | grep -E '^FAIL' | head -5
  { [ $rc -eq 0 ] && echo "$out" | grep -q '^SUMMARY.* 0 FAIL'; } || fail=1
done

if [ -n "${MODEL:-}" ]; then
  echo "== 3. model: $MODEL"
  P="The three primary colors are"
  g=$("$B/llama-completion" -m "$MODEL" -p "$P" -n 32 --temp 0 -c 1024 -ub 128 -ngl 99 -fa off -no-cnv 2>/dev/null | tr '\n' ' ')
  c=$("$B/llama-completion" -m "$MODEL" -p "$P" -n 32 --temp 0 -c 1024 -ngl 0 -fa off -no-cnv 2>/dev/null | tr '\n' ' ')
  echo "gpu: $g"; echo "cpu: $c"
  [ -n "$(echo "$g" | sed "s/$P//" | tr -d ' ')" ] || { echo "  no GPU text (does the model fit? try a smaller -c / -ub)"; fail=1; }
  "$B/llama-bench" -m "$MODEL" -ngl 99 -fa 0 -ub 128 -p 128 -n 64 -r 2 2>&1 | grep -E 'pp128|tg64'
fi

[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
