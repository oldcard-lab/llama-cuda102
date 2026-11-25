# llama-cuda102 (oldcard-lab)

Public-facing repository for the Kepler (sm_30) llama.cpp patch series. It is prepared for publishing by the owner:
nothing here may contain private details (host names, user paths, tracker URLs, credentials).

- `patches/`: `git format-patch` series on llama.cpp v1.0.0 (b1b770d9). Regenerate from the development repository
  (branch series/6uo-v1.0.0), never hand-edit a patch.
- `scripts/`: install-cuda-10.2.sh, build.sh, test.sh: standalone, generic paths only.
- `tools/mmvk-check.cpp`: strict kernel checker used by scripts/test.sh.
- `docs/`: RESULTS.md (numbers must come from a gated release), PROVENANCE.md, AI-USAGE.md.

Rules: the series must stay byte-identical in result to a gated release (see docs/PROVENANCE.md). Run the
private-detail scan before every commit (`scripts/scan-private.sh`, kept in the private development copy only: its
pattern list names private hosts).
Agents publish this repository to GitHub only through the gated push tool (`github_push.py` in the private workspace
tooling: reviewed and merged commits only, fast-forward to `main`, private paths removed, content scan, audit log),
never with a plain `git push`. A first publish to a new public repository is the owner's.
