#!/bin/bash
# Install the CUDA 10.2 TOOLKIT ONLY into a private prefix (no sudo, nothing system-wide).
#
# CUDA 10.2 is the last toolkit that supports compute capability 3.0 (sm_30). Its runfile bundles driver 440.33,
# which must NOT be run (keep your working 470 driver). The runfile's own installer may not run on current
# distributions (it needs libxml2.so.2), so the makeself payload is unpacked with --noexec (nothing executes) and
# the toolkit tree is copied into the prefix. No /usr/local/cuda symlink, no ld.so entry: the build points at the
# prefix and binaries get an rpath.
#
# Usage: scripts/install-cuda-10.2.sh [PREFIX]          (default: $HOME/cuda-10.2)
# Needs: about 6.5 GB free while unpacking (2.6 GB runfile + payload), 3.9 GB afterwards.
set -eu
PREFIX=${1:-$HOME/cuda-10.2}
WORK=${CUDA_WORK_DIR:-$HOME/.cache/cuda-10.2-install}
URL=https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
MD5=3d816c583759efa8567a92c1a2af90db   # NVIDIA's published md5 for this runfile (md5sum.txt next to it)
RUN=$WORK/cuda_10.2.89_440.33.01_linux.run

mkdir -p "$WORK"
if [ ! -f "$RUN" ]; then
  echo "downloading $URL (2.6 GB)"
  wget -O "$RUN.part" "$URL" && mv "$RUN.part" "$RUN"
fi
echo "$MD5  $RUN" | md5sum -c -
[ -d "$WORK/payload/builds/cuda-toolkit" ] || sh "$RUN" --noexec --keep --target "$WORK/payload"
if [ -e "$PREFIX" ]; then
  echo "$PREFIX exists; not overwriting it"
else
  cp -a "$WORK/payload/builds/cuda-toolkit" "$PREFIX"
  cp -a "$WORK/payload/builds/cublas/include/." "$PREFIX/targets/x86_64-linux/include/"
  cp -a "$WORK/payload/builds/cublas/lib64/." "$PREFIX/targets/x86_64-linux/lib/"
fi
cat "$PREFIX/version.txt"
ls "$PREFIX/lib64" | grep -E '^lib(cudart|cublas|cublasLt)\.so\.10' | head -3
echo "CUDA 10.2 toolkit in $PREFIX. You may now delete $WORK."
