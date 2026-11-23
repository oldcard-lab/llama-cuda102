# Provenance of the code in this series

## Why this document exists

This project began on top of a community patch posted in a llama.cpp GitHub discussion that adds old-GPU
(Fermi/Kepler) support. That patch states no licence. To publish under MIT without depending on it, the
compatibility layer in this series was **rewritten clean-room**, and the project's own kernels were ported onto the
clean layer. The earlier, community-based line of development is not part of this repository.

## What is in each patch

| Patch | Content | Origin |
|---|---|---|
| 0001 | sm_30 / CUDA 10.2 / clang build compatibility | clean-room (job A) |
| 0002 | sm_30 / CUDA 10.2 runtime compatibility | clean-room (job B) |
| 0003 | Kepler matvec and attention kernels, prompt path, dispatch | this project's own work, plus job C build fixes |
| 0004 | Kepler im2col | this project's own work |
| 0005 | FunctionGemma tool-call parser | this project's own work |

## Clean-room procedure (patches 0001-0002, and the job C fixes in 0003)

- The layer was written by AI coding agents (OpenAI Codex, model gpt-6-luna) in a **new repository whose only
  history is the unmodified upstream release** (llama.cpp v0.8.0, verified byte-identical to upstream).
- Each agent's instructions contained only the toolchain, the goal and the tests, and explicitly forbade opening
  the earlier project tree, its notes and patches, the community discussion and any other old-GPU fork.
- The agents worked from compiler errors, test output and NVIDIA's CUDA 10.2 headers and documentation. Each change
  is recorded with its reason (the error or failing test that required it).
- The coordinating engineer, who had read the community patch, wrote no compatibility code; compiler and test output
  were routed back to the agents. Diagnostics the coordinator ran (running a model to see why it produced no text)
  produced output only.
- **Audits after each job:**
  - every command each agent ran was checked for access to the forbidden paths (none found);
  - a mechanical overlap check compared the non-trivial added lines of the clean layer with those of the community
    patch.
- **Overlap result:** 10 of 121 lines are identical, and the CUDA API dictates all of them.
  - Seven add the `0` flags argument that CUDA 10.2's `cudaStreamWaitEvent` requires to existing upstream calls.
  - Two are the standard `#if CUDART_VERSION >= 11000` / `#include <cuda_bf16.h>` guard.
  - One is a one-line type alias (`#define CUDA_R_16BF CUDA_R_16F`).
  - The job C fixes share no lines with the community patch.
- **Acceptance:**
  - the clean layer alone passes llama.cpp's full `test-backend-ops` suite on the GPU, with the same case count as
    the earlier build;
  - with the kernels on top, the full release gate passes at identical speed (see RESULTS.md).

## Hunks of the project's own work that were not carried over

When the project's own changes were ported onto the clean layer, any change that edited or sat inside code from the
community patch was not copied. Most of those changes only adjusted the community layer itself and are unnecessary
on the clean layer. The two that implemented features were re-written by a clean-room agent from a functional
description:
- F32 output for F16 compute on GPUs without fast FP16;
- batched SGEMM below compute capability 5.0.

## Third-party components

- llama.cpp / ggml (MIT): the upstream code this series patches; see NOTICE.
- The test tool `tools/mmvk-check.cpp` is this project's own; its approach follows Wafer.ai's published guards
  against reward-hacked AI-written kernels (no code taken).
- No model weights, CUDA, cuDNN or other binaries are distributed here.
