# Provenance of the code in this series

## What is in each patch

| Patch | Content | Origin |
|---|---|---|
| 0001 | 6uO's old-GPU CUDA compatibility port, forward-ported to llama.cpp v0.8.0 and then v1.0.0 | 6uO, MIT |
| 0002 | Fixes to the port: BF16 shim, graph compute without CUDA graphs, F32 output rule, and classic FP32 cuBLAS GEMMs below compute capability 5.0 | this project |
| 0003 | Kepler matrix-vector and attention kernels, prompt path, and dispatch | this project |
| 0004 | Kepler im2col path | this project |
| 0005 | FunctionGemma tool-call parser | this project |

## How the port was forward-ported

6uO released the original old-GPU CUDA port under the MIT license on 2026-11-23. It targeted ggml 0.25.3; this
series adapts it to llama.cpp v0.8.0 (ggml 0.28.0). Of 142 hunks, 123 applied at fuzz 2 or less. OpenAI Codex
agents adapted the 19 rejected hunks to the newer code. Patch 0001 retains 6uO as its Git author. An Anthropic
Claude agent debugged the forward port and made the fixes in patch 0002.

On 2026-11-24 the series moved from v0.8.0 to llama.cpp v1.0.0 (commit b1b770d, ggml 1.0.0). Patch 0001 had 10
conflicting hunks in 6 files, all where upstream had changed the same lines (mostly its new FP8 support); 0002 and
0003 had one conflict each in `ggml-cuda.cu`; 0004 and 0005 applied unchanged. A Codex agent resolved the 0001
conflicts, a Claude agent resolved the others and fixed what the first build and two independent reviews (Muse Code,
Gemini 4 Neon through AGY) found: an FP8 branch in `mmvf.cu` placed behind a CUDA 11 guard, and FP8 weights missing
from the low-memory conversion chunking for older cards (both checked afterwards with an FP8 model; see RESULTS). A test after the gate found FunctionGemma tool calls
returning empty arguments: v1.0.0's shared Gemma 4 parser now finds argument nodes by tag instead of rule name, so
patch 0005 tags them the same way (the release gate now checks tool-call arguments). These forward-port resolutions are this project's, not
6uO's; the Git author of 0001 stays 6uO because the port itself is 6uO's.

Applied with `git am --keep-cr` to upstream's v1.0.0 tag, the series produces exactly the source of the v1r3
release on which the gated results in RESULTS.md were measured (the v1.0.0 snapshot used for development has the
same tree as the tag). The v1r3 gate also matched the earlier v0.8.0 release (r29): same perplexity on all 29
gated models, speed within 1%. A build from this repository on the test card passed the same checks: llama.cpp's
GPU op tests, the strict kernel checker, and matching speed.

## History

The original port's license was unknown, so patches 0001-0002 were first developed as a clean-room rewrite. After
6uO granted the port under MIT, the owner chose to build on 6uO's port, the base this project had been developed
and tested on from the start, and retired the rewrite. The rewrite remains in this repository's Git history.

## Third-party components

- 6uO's port in patch 0001: MIT; see NOTICE.
- llama.cpp / ggml (MIT): the upstream code this series patches; see NOTICE.
- The test tool `tools/mmvk-check.cpp` is this project's own; its approach follows Wafer.ai's published guards
  against reward-hacked AI-written kernels (no code taken).
- No model weights, CUDA, cuDNN or other binaries are distributed here.
