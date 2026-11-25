# llama-cuda102: llama.cpp on NVIDIA Kepler (sm_30) GPUs with CUDA 10.2

Part of [oldcard-lab](https://github.com/oldcard-lab), experiments with running current AI models on old graphics
cards. One repository per toolchain: this one covers cards that need CUDA 10.2.

A patch series for [llama.cpp](https://github.com/ggml-org/llama.cpp) that runs its CUDA backend on Kepler GPUs
with compute capability 3.0 (GeForce GTX 600 / 700 series cards such as the GTX 650 Ti, 660, 670, 680, 760, 770),
plus Kepler-tuned kernels that make small language models, speech and vision models practical on a 1-2 GB card.

Current CUDA releases dropped Kepler long ago. This series builds llama.cpp's CUDA backend with the last toolkit
that supports sm_30 (CUDA 10.2), using clang as the CUDA compiler, on top of the old-GPU CUDA port by 6uO (see
"Credits" below), and adds:

- **Kepler matrix-vector kernels** for token generation (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, Q5_0, IQ3_XXS, F16) and the
  attention matvecs, 1.8-3.3x the stock CUDA kernels on this hardware.
- **A faster prompt path** on Kepler: row-chunked dequantization and token columns padded so cuBLAS 10.2 picks its
  fast sm_30 GEMM kernel.
- **A Kepler im2col** for convolution models (YOLO-class detectors), 18% faster per frame.
- A **FunctionGemma** tool-call parser for `llama-server --jinja`.

Measured on a GeForce GTX 650 Ti (GK106, 1 GB) with an i7-4770: see [docs/RESULTS.md](docs/RESULTS.md).
Examples: LFM2.5-350M Q4_K_M generates 88.0 tokens/s; Qwen3-1.7B (3-bit) 23.2 tokens/s; whisper small.en
transcribes an 11 s clip in about 2.6 s; YOLOv3-tiny runs at 26.8 frames/s.

## Requirements

- An NVIDIA GPU with compute capability 3.0 (see "Tested on" below).
- Linux x86-64 with the NVIDIA **470** driver (the last branch that supports Kepler); see "Driver setup" below.
  Tested on Ubuntu 26.04 (kernel 7.0) with 470.256.02 built as a DKMS module.
- The **CUDA 10.2 toolkit** (toolkit only; do not install its bundled 440 driver): `scripts/install-cuda-10.2.sh`
  installs it into a private prefix without root.
- **clang 22** (clang-22, clang++-22, ld.lld-22), cmake, ninja, git. CUDA 10.2's nvcc only accepts old host
  compilers, so clang compiles the CUDA code. On Ubuntu 26.04: `sudo apt install clang-22 lld-22 cmake ninja-build
  git`. On older distributions, get clang 22 from the LLVM project's packages (https://apt.llvm.org).

## Driver setup

Kepler's last driver is **470.256.02**. Getting it running is usually the hardest step, and how hard depends on your
kernel:

- **Older kernels:** if your distribution still packages the 470 driver for its kernel (for example
  `nvidia-driver-470` on Ubuntu 22.04), install that package and reboot.
- **Kernel 7.0 (Ubuntu 26.04, current Arch):** NVIDIA's 470.256.02 installer does not build on this kernel by
  itself. We use the community patches from
  [haxllo/nvidia-470xx-linux-7.0](https://github.com/haxllo/nvidia-470xx-linux-7.0) (GPL-2.0, not part of this
  repository); follow its README. Our setup: NVIDIA's `.run` installer for the user-space parts, the patched kernel
  modules registered with DKMS so that kernel updates rebuild them.
- **Secure Boot:** the kernel modules must be signed. With Ubuntu's DKMS, enroll its signing key once
  (`sudo mokutil --import /var/lib/shim-signed/mok/MOK.der`, reboot, confirm in the blue MOK screen), or turn Secure
  Boot off.

Check before building: `cat /proc/driver/nvidia/version` should report 470.256.02, and `nvidia-smi` should list the
card. The 470 driver supports CUDA up to 11.4, so the CUDA 10.2 toolkit runs on it without its own driver.

## Build

```bash
scripts/install-cuda-10.2.sh ~/cuda-10.2        # once: downloads 2.6 GB from NVIDIA, verifies the md5
scripts/build.sh work ~/cuda-10.2                # clones llama.cpp v1.0.0, applies patches/, builds
scripts/test.sh  work ~/cuda-10.2                # correctness checks on the GPU (MODEL=x.gguf adds a model run)
```

The binaries are in `work/llama.cpp/build/bin` (`llama-cli`, `llama-server`, `llama-completion`, `llama-bench`).
The series applies to llama.cpp **v1.0.0** (commit `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`) with `git am --keep-cr`
(one file in patch 0003 has CRLF line endings; `scripts/build.sh` does this for you).

## Using it

Run models fully on the GPU (`-ngl 99`) and keep flash attention off (`-fa off`: this build has no flash-attention
kernels for sm_30; "auto" costs about 8%). A 1 GB card fits models up to about 900 MB with a modest context:

```bash
llama-server -m LFM2.5-350M-Q4_K_M.gguf -ngl 99 -fa off -t 1
llama-cli    -m Qwen3-1.7B-Q3_K_S.gguf  -ngl 99 -fa off -c 1024 -ub 32
```

- Models of about 700 MB and up need a smaller micro-batch: `-ub 128`, or `-ub 32` for 1.7B models at 3 bits.
- 1.7B models fit at 3 bits per weight (Q3_K_S / Q3_K_M with an importance matrix keep quality close to 4-bit).
- `-ctk q8_0` (8-bit K cache) frees enough memory for `-ub 64` at `-c 1024` on 1.7B models: prompt processing
  57 t/s instead of 31, for about +1.2% perplexity and -3% generation speed.
- Every Kepler path has a switch: `GGML_CUDA_KEPLER_MMV=0` (matvec kernels off), `GGML_CUDA_KEPLER_PP=off`,
  `GGML_CUDA_KEPLER_IM2COL=off` restore the stock code paths; `GGML_CUDA_KEPLER_TRACE=1` prints which kernel ran.

## Tested on

| Card | Compute capability | Status |
|---|---|---|
| GeForce GTX 650 Ti (GK106, 1 GB) | 3.0 | Tested: every release gated on it |
| Other sm_30 cards (GTX 660 / 670 / 680 / 760 / 770, ...) | 3.0 | Should work, untested |
| GTX 780 / Titan, GT 710 / 730 (GK208), Tesla K20 / K40 | 3.5 | Not built for yet (CUDA 10.2 supports it; untested) |
| Tesla K80 | 3.7 | Not targeted |
| Maxwell (GTX 750 Ti, 900 series) | 5.0 / 5.2 | Not built for yet (CUDA 10.2 supports it; untested) |

Reports from other cards are welcome as issues. For sm_35 / sm_37 today, see
[babal35/llamacpp-kepler](https://github.com/babal35/llamacpp-kepler) (CUDA 11.x build fixes) and
[theIvanR/llama-on-legacy-gpu](https://github.com/theIvanR/llama-on-legacy-gpu) (compute capability 3.5-6.2).

## Limits

- BF16 operations run on the CPU (CUDA 10.2 has no BF16 support).
- No flash attention; no CUDA graphs.
- The Kepler kernels and the prompt path only run below compute capability 5.0. The compatibility port (patches
  0001-0002) targets old CUDA toolkits and old GPUs; it has been tested only on the GTX 650 Ti, so this build is not
  meant for newer cards.

## Credits

- **6uO**: the old-GPU CUDA port this series is built on (patch 0001), from llama.cpp discussion
  [#29961](https://github.com/ggml-org/llama.cpp/discussions/29961), released under the MIT license.
  Forward-ported to v0.8.0 and then to v1.0.0 here.
- **This project (oldcard-lab)**: the fixes to the port (0002), the Kepler kernels and prompt path (0003), the
  im2col path (0004), the FunctionGemma parser (0005), the scripts and the strict kernel checker.
- **The ggml authors**: llama.cpp and ggml themselves (MIT).

Details per patch: [docs/PROVENANCE.md](docs/PROVENANCE.md). The
[kepler-llama-community-patch](https://github.com/benaiahehud/kepler-llama-community-patch) repository is this
project's earlier development line: the same patches applied to a full llama.cpp tree. It is also where 6uO's
further ports may land.

## How it was made: mostly by AI

AI coding agents did most of the engineering here, arguably all of the heavy lifting: the kernels, the forward port of
6uO's compatibility layer, the test tools and these documents. A human owner set the goals, provided the hardware
and approved each direction. Details, and the safeguards that do not depend on trusting the agents, are in
[docs/AI-USAGE.md](docs/AI-USAGE.md). The strict kernel checks follow Wafer's
[field guide to reward hacking in AI kernel generation](https://www.wafer.ai/blog/reward-hacks-field-guide).

## Security

Pinned to llama.cpp v1.0.0 and end-of-life CUDA 10.2 / driver 470: read [SECURITY.md](SECURITY.md) first.

## License

MIT, like llama.cpp. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
