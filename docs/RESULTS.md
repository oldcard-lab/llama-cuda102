# Results

Hardware: GeForce GTX 650 Ti (GK106, compute capability 3.0, 4 SMX, 1 GB GDDR5, ~86 GB/s), Intel i7-4770,
NVIDIA 470.256.02, CUDA 10.2 toolkit, clang 22. Build: this patch series on llama.cpp v1.0.0. Correctness and token
generation are from the v1r3 release gate (2026-11-25), whose source this series reproduces exactly; so are the
"This build" prompt numbers at the gate's micro-batch sizes (32, 64 and 128 below; the gate does not run -ub 64 on
Qwen3-1.7B). On every gated model the v1.0.0 build gives the same perplexity as the v0.8.0
build (r29) and the same speed within 1%. The other sections were measured on earlier builds of the same kernels on
v0.8.0 (2026-11-22). `-ngl 99 -fa off -t 1`.

## Correctness (release gate)

- `test-backend-ops`, every op, CUDA vs CPU: 15145 OK, 0 FAIL (4 cases too large for 1 GB are skipped).
- `tools/mmvk-check` (per-row fp64 reference, NaN-poisoned output, bitwise repeat, weights-buffer mode): 597 cases,
  0 FAIL, worst relative error 1.6e-7 (limit 2e-5), both modes.
- Every model below: perplexity equal to the expected value, greedy text sane, no GPU faults (Xid) during the gate.
- FP8 (`F8_E4M3`) weights, outside the gate (2026-11-25): LFM2.5-350M with every weight matrix set to `f8_e4m3` (requantized from Q8_0; norms stay F32) gives the
  same perplexity on the card as on the CPU (141.16 vs 141.19 batched at `-ub 128`, which goes through the
  row-chunked cuBLAS conversion; 43.74 vs 43.75 at `-ub 1`, the FP8 matrix-vector kernel), with no GPU faults. It is slow here: 17 t/s generation
  against 88 for Q4_K_M, because Kepler has no FP8 hardware and no tuned kernel for it. Use Q4_K/Q8_0 instead.

## Token generation (tg128, tokens/s)

| Model | Quant | t/s |
|---|---|---:|
| LFM2.5-350M | Q4_K_M | 88.9 |
| gemma-3-270m-it | Q4_K_M | 64.6 |
| FunctionGemma-270M | Q8_0 | 65.0 |
| LFM2.5-350M | Q8_0 | 59.6 |
| granite-4.0-350m | Q8_0 | 47.9 |
| Qwen3-0.6B | Q4_K_M | 45.8 |
| SmolLM2-360M-Instruct | Q8_0 | 44.8 |
| Qwen2.5-0.5B-Instruct | Q8_0 | 38.8 |
| Qwen2.5-Coder-0.5B-Instruct | Q8_0 | 38.8 |
| Qwen3.5-0.8B | Q4_K_M | 35.6 |
| TinyLlama-1.1B-Chat | Q4_K_M | 34.8 |
| Qwen3-0.6B | Q8_0 | 32.6 |
| LFM2.5-1.2B-Instruct | Q4_K_M | 32.6 |
| LFM2-1.2B-Tool | Q4_K_M | 32.6 |
| Qwen3-0.6B | Q5_K_M | 30.0 |
| Llama-3.2-1B-Instruct | Q4_K_M | 29.0 |
| OLMo-2-0425-1B-Instruct | Q4_K_S | 28.1 |
| Qwen2.5-Coder-1.5B-Instruct | Q3_K_S (imatrix) | 23.8 |
| Qwen2.5-1.5B-Instruct | Q3_K_S (imatrix) | 23.8 |
| SmolLM2-1.7B-Instruct | Q3_K_S (imatrix) | 23.5 |
| Qwen3-1.7B | Q3_K_S (imatrix) | 23.2 |
| granite-4.0-h-1b | Q3_K_S (imatrix) | 22.8 |
| LFM3-1.7B | Q3_K_M (imatrix) | 22.2 |
| granite-4.0-1b | Q3_K_S (imatrix) | 22.2 |
| NuExtract-2.0-2B | Q3_K_M | 22.2 |
| gemma-3-1b-it | Q4_K_M | 20.1 |
| LFM3-1.7B | IQ3_XXS | 13.4 |

### Kepler kernels vs the stock CUDA kernels (same build, `GGML_CUDA_KEPLER_MMV=0` for stock; tg64, t/s)

| Model | Stock | Kepler | Gain |
|---|---:|---:|---:|
| LFM2.5-350M Q8_0 | 32.8 | 59.6 | 1.8x |
| Qwen3-0.6B Q8_0 | 18.6 | 32.7 | 1.8x |
| Qwen2.5-Coder-0.5B Q8_0 | 26.7 | 38.7 | 1.4x |
| Qwen3-1.7B Q3_K_S | 7.1 | 23.2 | 3.3x |

## Prompt processing (llama-bench pp256, tokens/s)

| Model | -ub | Stock path (`GGML_CUDA_KEPLER_PP=off`) | This build |
|---|---:|---:|---:|
| Qwen3-1.7B Q3_K_S | 32 | 26.7 | 32.5 |
| Qwen3-1.7B Q3_K_S | 64 | 29.3 | 63.0 |
| Qwen2.5-Coder-1.5B Q3_K_S | 32 | 26.2 | 32.4 |
| LFM2.5-1.2B Q4_K_M | 64 | 41.6 | 95.4 |
| Qwen3-0.6B Q8_0 | 128 | 321.6 | 321.8 |
| LFM2.5-350M Q4_K_M | 128 | 615.8 | 616.2 |

## Agent-style calls (prompt of N tokens + 128 generated, modelled from measured pp and tg at depth)

| Model | -ub | Call | Seconds |
|---|---:|---|---:|
| LFM2.5-350M Q4_K_M | 128 | 2048 + 128 | 5.3 |
| Qwen3-0.6B Q8_0 | 128 | 1024 + 128 | 8.6 |
| LFM2.5-1.2B Q4_K_M | 64 | 2048 + 128 | 26.5 |
| Qwen3-1.7B Q3_K_S | 32 | 768 + 128 | 30.2 |
| Qwen2.5-Coder-1.5B Q3_K_S | 32 | 1024 + 128 | 38.2 |
| LFM3-1.7B Q3_K_M | 32 | 2048 + 128 | 73.0 |

## Speech (whisper.cpp 1.9.5 built against this ggml, `-nfa -bs 1 -bo 1`, 11 s clip)

| Model | Wall time |
|---|---:|
| base.en q8_0 | 1.2-1.4 s |
| small.en q8_0 | 2.6-3.0 s |

The v1r3 gate measured base.en 1.32 s and small.en 3.05 s. (CPU build on the same machine: small.en 5.4 s. Whisper runs single-threaded here, so its wall time moves with
other CPU load.)

## Vision (YOLO-class detection, 416x416, one stream, inference only)

| Detector | This build (ggml, GPU) | OpenCV 4.12 + cuDNN 7.6.5 (GPU) | OpenVINO 2026.4, 3 CPU cores |
|---|---:|---:|---:|
| YOLOv3-tiny | 26.8 fps | 54 fps | - |
| YOLOv4-tiny | - | 42 fps | - |
| YOLOX-Tiny | 15.6 fps (before the im2col patch) | 17.6 fps | 21.7 fps |
| YOLOX-Nano | 33.0 fps (before the im2col patch) | 33.9 fps | 73 fps |

The ggml YOLO numbers use ggml's `examples/yolo` with batch norm folded into the weights and a 32-bit im2col graph;
the YOLOX ggml graph is a separate port (not part of this patch series). cuDNN is proprietary and is not part of
this project; it is listed as a comparison.

## Windows 11 parity (same GTX 650 Ti, driver 456.71 WDDM, patch 0006)

Measured on the same GeForce GTX 650 Ti and i7-4770 booted into Windows 11 Pro x64 (driver 456.71 WDDM, CUDA 10.2.89,
clang 22.1.8, `-DGGML_CUDA_GRAPHS=ON`). Under WDDM with the card hosting the desktop session, Windows uses 120-125 MiB
of VRAM at idle (~900 MiB free vs ~1024 MiB on headless Linux); every model and workload that fits in ~900 MiB passes
with zero `nvlddmkm` or Display 4101 TDR events:

- **Correctness:** `test-backend-ops` ALL ops: `15145 OK, 0 FAIL` (same 4 cases too large for 1 GB as Linux);
  `tools/mmvk-check`: `597/597` plain (worst relative error `1.21e-7`) and `597/597` `--weights` (`1.88e-7`); all 18
  gated models that fit in ~900 MiB match Linux perplexity to 4 decimal places (`0.0000%-0.005%` difference) and match
  `-ub 1` decode vs batched perplexity within `0.02%`.
- **Token generation (`tg128`, tokens/s):** LFM2.5-350M Q4_K_M `84.9` (`tg64 86.6`, up from `27.9` before patch 0006;
  Linux `88.9`), LFM2.5-350M Q8_0 `58.2` (Linux `59.6`), Qwen3-0.6B Q4_K_M `44.9` (Linux `45.8`), Qwen3-0.6B Q8_0
  `32.3` (Linux `32.6`), Qwen3.5-0.8B Q4_K_M `35.0` (Linux `35.6`), LFM2.5-1.2B Q4_K_M `32.5` (Linux `32.6`),
  TinyLlama-1.1B Q4_K_M `34.5` (Linux `34.8`), SmolLM2-360M Q8_0 `43.8` (Linux `44.8`), granite-4.0-350m Q8_0 `46.8`
  (Linux `47.9`), Qwen2.5-Coder-1.5B Q3_K_S `23.4` (Linux `23.8`), granite-4.0-h-1b Q3_K_S `22.4` (Linux `22.8`).
- **Prompt processing (`pp256`, tokens/s):** LFM2.5-350M Q4_K_M `-ub 128`: `615.4` (Linux `616.2`); Qwen3-0.6B Q8_0
  `-ub 128`: `320.6` (Linux `321.8`); LFM2.5-1.2B Q4_K_M `-ub 64`: `94.9` (Linux `95.4`); Qwen2.5-Coder-1.5B Q3_K_S
  `-ub 32`: `32.2` (Linux `32.4`).
- **Agent-style calls:** LFM2.5-350M `2048 + 128`: `5.4 s` (Linux `5.3 s`); Qwen2.5-Coder-1.5B `1024 + 128`: `38.6 s`
  (Linux `38.2 s`).
- **FP8 (`F8_E4M3`):** LFM2.5-350M GPU vs CPU perplexity `141.1579 vs 141.1882` (`0.0215%` at `-ub 128`) and
  `43.7412 vs 43.7489` (`0.0176%` at `-ub 1`); `pp128 642.2 t/s`, `tg64 17.4 t/s`.
- **Speech (whisper.cpp 1.9.5 built against this ggml, 11 s clip):** `base.en q8_0` `1.31 s` (Linux `1.32 s`),
  `small.en q8_0` `2.70 s` (Linux `3.05 s`).
- **Tool calls, embeddings, reranking, vision, speculative & server soak:** FunctionGemma 4/4 tool-call prompts,
  granite-4.0-350m `--jinja`, `embeddinggemma-300M`, `bge-reranker-v2-m3`, VLM (`SmolVLM2-256M`, `granite-docling-258M`,
  `LFM2.5-VL-450M` at `-c 2048`), `yolox-ggml` (`max abs 0` vs reference), OpenCV 4.12 + cuDNN 7.6.5 YOLOX (identical
  detections to Linux/CPU), n-gram speculative decoding on `Qwen2.5-Coder-0.5B` (`35.4 -> 51.6 t/s`, +46%, bit-identical
  output), and 72-request concurrent `llama-server` soak (`-np 1/2/4`, 72/72 OK) all pass with zero GPU faults.
- **WDDM VRAM limit note:** 1.7B/2B models and `-c 4096` 450M-500M VLMs that fit on headless Linux OOM on CUDA under
  120-125 MiB WDDM desktop reservation unless an integrated GPU hosts the Windows desktop.

## Negative results (kept so nobody repeats them)

- Texture-cache loads of the activation vector at 1 token: no gain. More columns per launch than 4-8: slower.
- ggml's direct convolution (`ggml_conv_2d_direct`) on this card: 2.6 fps on YOLOv3-tiny (unusable).
- Batching camera frames: no per-stream gain on this card.
- Speculative decoding (n-gram) helps 8-bit 0.5B models on edit-style output (+35%) but slows 3-bit 1.7B models.
