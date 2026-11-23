# Results

Hardware: GeForce GTX 650 Ti (GK106, compute capability 3.0, 4 SMX, 1 GB GDDR5, ~86 GB/s), Intel i7-4770,
NVIDIA 470.256.02, CUDA 10.2 toolkit, clang 22. Build: this patch series on llama.cpp v0.8.0. All numbers from the
release gate and workload benchmark of that build (2026-11-22) unless noted. `-ngl 99 -fa off -t 1`.

## Correctness (release gate)

- `test-backend-ops`, every op, CUDA vs CPU: 14361 OK, 0 FAIL (4 cases too large for 1 GB are skipped).
- `tools/mmvk-check` (per-row fp64 reference, NaN-poisoned output, bitwise repeat, weights-buffer mode): 597 cases,
  0 FAIL, worst relative error 1.6e-7 (limit 2e-5), both modes.
- Every model below: perplexity equal to the expected value, greedy text sane, no GPU faults (Xid) during the gate.

## Token generation (tg128, tokens/s)

| Model | Quant | t/s |
|---|---|---:|
| LFM2.5-350M | Q4_K_M | 88.6 |
| gemma-3-270m-it | Q4_K_M | 64.6 |
| FunctionGemma-270M | Q8_0 | 64.9 |
| LFM2.5-350M | Q8_0 | 59.6 |
| granite-4.0-350m | Q8_0 | 47.8 |
| Qwen3-0.6B | Q4_K_M | 45.7 |
| SmolLM2-360M-Instruct | Q8_0 | 44.7 |
| Qwen2.5-0.5B-Instruct | Q8_0 | 38.7 |
| Qwen2.5-Coder-0.5B-Instruct | Q8_0 | 38.7 |
| Qwen3.5-0.8B | Q4_K_M | 35.5 |
| TinyLlama-1.1B-Chat | Q4_K_M | 34.7 |
| Qwen3-0.6B | Q8_0 | 32.6 |
| LFM2.5-1.2B-Instruct | Q4_K_M | 32.5 |
| LFM2-1.2B-Tool | Q4_K_M | 32.5 |
| Qwen3-0.6B | Q5_K_M | 29.9 |
| Llama-3.2-1B-Instruct | Q4_K_M | 28.9 |
| OLMo-2-0425-1B-Instruct | Q4_K_S | 28.1 |
| Qwen2.5-Coder-1.5B-Instruct | Q3_K_S (imatrix) | 23.8 |
| Qwen2.5-1.5B-Instruct | Q3_K_S (imatrix) | 23.8 |
| SmolLM2-1.7B-Instruct | Q3_K_S (imatrix) | 23.5 |
| Qwen3-1.7B | Q3_K_S (imatrix) | 23.2 |
| granite-4.0-h-1b | Q3_K_S (imatrix) | 22.9 |
| LFM3-1.7B | Q3_K_M (imatrix) | 22.2 |
| granite-4.0-1b | Q3_K_S (imatrix) | 22.2 |
| NuExtract-2.0-2B | Q3_K_M | 22.1 |
| gemma-3-1b-it | Q4_K_M | 20.2 |
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
| Qwen3-0.6B Q8_0 | 128 | 321.6 | 321.6 |
| LFM2.5-350M Q4_K_M | 128 | 615.8 | 615.7 |

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

(CPU build on the same machine: small.en 5.4 s. Whisper runs single-threaded here, so its wall time moves with
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

## Negative results (kept so nobody repeats them)

- Texture-cache loads of the activation vector at 1 token: no gain. More columns per launch than 4-8: slower.
- ggml's direct convolution (`ggml_conv_2d_direct`) on this card: 2.6 fps on YOLOv3-tiny (unusable).
- Batching camera frames: no per-stream gain on this card.
- Speculative decoding (n-gram) helps 8-bit 0.5B models on edit-style output (+35%) but slows 3-bit 1.7B models.
