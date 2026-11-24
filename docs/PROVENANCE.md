# Provenance of the code in this series

## What is in each patch

| Patch | Content | Origin |
|---|---|---|
| 0001 | 6uO's old-GPU CUDA compatibility port, forward-ported to llama.cpp v0.8.0 | 6uO, MIT |
| 0002 | v0.8.0 fixes: BF16 shim, graph compute without CUDA graphs, F32 output rule, and classic FP32 cuBLAS GEMMs below compute capability 5.0 | this project |
| 0003 | Kepler matrix-vector and attention kernels, prompt path, and dispatch | this project |
| 0004 | Kepler im2col path | this project |
| 0005 | FunctionGemma tool-call parser | this project |

## How the port was forward-ported

6uO released the original old-GPU CUDA port under the MIT license on 2026-11-23. It targeted ggml 0.25.3; this
series adapts it to llama.cpp v0.8.0 (ggml 0.28.0). Of 142 hunks, 123 applied at fuzz 2 or less. OpenAI Codex
agents adapted the 19 rejected hunks to the newer code. Patch 0001 retains 6uO as its Git author. An Anthropic
Claude agent debugged the forward port and made the fixes in patch 0002.

The resulting source tree is byte-identical to the r29 release on which every published result in RESULTS.md was
measured, so those results are unchanged.

## History

The original port's license was unknown, so patches 0001-0002 were first developed as a clean-room rewrite. After
6uO granted the port under MIT, the owner chose to build on 6uO's work, which was developed and tested on the card
from the start, and retired the rewrite. The rewrite remains in this repository's Git history.

## Third-party components

- 6uO's port in patch 0001: MIT; see NOTICE.
- llama.cpp / ggml (MIT): the upstream code this series patches; see NOTICE.
- The test tool `tools/mmvk-check.cpp` is this project's own; its approach follows Wafer.ai's published guards
  against reward-hacked AI-written kernels (no code taken).
- No model weights, CUDA, cuDNN or other binaries are distributed here.
