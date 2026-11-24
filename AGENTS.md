# llama-cuda102 (oldcard-lab)

Public-facing repository for the Kepler (sm_30) llama.cpp patch series. It is prepared for publishing by the owner:
nothing here may contain private details (host names, user paths, tracker URLs, credentials).

- `patches/`: `git format-patch` series on llama.cpp v0.8.0 (8642541b). Regenerate from the clean development repo
  (branch publish/series), never hand-edit a patch.
- `scripts/`: install-cuda-10.2.sh, build.sh, test.sh: standalone, generic paths only.
- `tools/mmvk-check.cpp`: strict kernel checker used by scripts/test.sh.
- `docs/`: RESULTS.md (numbers must come from a gated release), PROVENANCE.md, AI-USAGE.md.

Rules: only the clean line goes here (see docs/PROVENANCE.md). Run the private-detail scan before every commit
(`scripts/scan-private.sh`, kept in the private development copy only: its pattern list names private hosts).
Agents do not push this repository to any public host; the owner does.
