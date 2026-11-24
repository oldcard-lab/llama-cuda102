# How AI was used

llama.cpp asks contributors to disclose AI use and to understand what they submit. This project was built with AI
coding agents under human direction. In detail:

- **Kernels** (`mmvk.cu`, the prompt path, im2col): written and iterated by AI coding agents (Google's Gemini via
  AGY, OpenAI Codex), each round driven by a written design brief and measured on the real card.
- **Compatibility layer** (patch 0001): 6uO's MIT-licensed port (discussion #29961), forward-ported by OpenAI
  Codex agents and debugged by the Anthropic Claude agent; the Claude agent also made the fixes in patch 0002.
- **Coordination, test design and release engineering:** an Anthropic Claude agent. That covered the test
  pipeline, the strict checker, the release gate, benchmarks and these documents.
- **Reviews:** independent reviews by other models (Gemini, GPT, Claude, Muse) at design points; their findings and
  what was adopted are kept in the project's notes.
- **Human role:** the owner set the goals and the hardware, approved every design direction and release policy, and
  writes every public post. Agents do not post publicly.

Safeguards that do not depend on trusting the agents:

- **A strict correctness tool** (`tools/mmvk-check.cpp`) checks every row of every kernel against an fp64
  reference. It also poisons outputs with NaN and requires bitwise-repeatable results. That catches kernels that
  game the usual average-error tests.
- **A release gate.** Every release that went live passed llama.cpp's full op suite on the GPU, the strict tool,
  perplexity and speed floors on 28 models, and a whisper transcription check, with GPU fault counters watched.
- **Negative results are recorded** (RESULTS.md), and estimates are never reported as measurements.
