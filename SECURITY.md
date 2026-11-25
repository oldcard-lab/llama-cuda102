# Security

This project makes llama.cpp run fast on old NVIDIA Kepler GPUs (sm_30, e.g. the GTX 650 Ti). Supporting that
hardware means pinning old software, so read this before you run it anywhere that matters.

## What to know

- **It is frozen on llama.cpp v1.0.0** (upstream commit b1b770d, 2026-11-13). Security fixes that upstream llama.cpp released
  after that version are not included. llama.cpp has published security advisories in the past (for example in
  model-file parsing and in `llama-server`). We have not checked which of them apply to v1.0.0. See
  https://github.com/ggml-org/llama.cpp/security for upstream's advisories.
- **It depends on end-of-life NVIDIA software.** CUDA 10.2 and the NVIDIA 470 legacy driver no longer receive
  security updates. That is the price of supporting Kepler cards.
- **The Kepler kernels were tested for correct results, not for security.** They have not been reviewed for
  hostile input. Only load model files (GGUF) from sources you trust.
- **Do not expose `llama-server` to the internet.** Keep it on localhost or a private network you control, and put
  authentication in front of it if others can reach it.

## Reporting a problem

Open an issue in this repository. If the problem is sensitive (for example, a way to run code through a crafted
model file), open an issue asking for a private contact without the details, and we will follow up.

Problems that also exist in upstream llama.cpp should be reported to upstream as well:
https://github.com/ggml-org/llama.cpp/security
