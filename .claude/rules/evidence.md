---
paths:
  - "docs/**/*.md"
  - "CHANGELOG.md"
  - "ROADMAP.md"
  - "README.md"
  - "bench/**"
  - "paper/**"
---

# Evidence rules for xllama docs (local; PR #511 review, 2026-11-23)

- No performance figure in the repo (tok/s, ratios, µs/ms, MB deltas, "×N",
  B/A, intervals) without a `bench/results/<date>-<slug>/` directory in the
  same PR, cited in the same paragraph. "Operator-declared, not project
  evidence" in docs is NOT enough. Series X numbers go into the issue/PR
  body, labelled operator-declared.
- What may stay in docs without data: submission/dispatch/layer counts and a
  narrowed parity sentence that points at where the data lives. Patterns:
  `origin/main:docs/gguf-gpu-decode.md:1014` ("Measured on Series X: Bielik
  421 → 301 submissions per token") and `origin/main:CHANGELOG.md:18`
  ("Series X comparisons reported in #503 matched logits ...").
- Never cite a figure that is neither on `origin/main` nor in committed data,
  and build no estimate on such a figure (the "×1.107" case).
- Parity: a `bench/results/` artifact or a `#N` that holds the comparison. In
  a PR body: "(operator-declared, Series X)" + models + ubatch. Gate and docs
  state the same contract: an NMSE bound is not "bit for bit".
- Any sentence about a fallback, a refusal, or "without the file nothing
  changes": list every failure path with its code line; the sentence must be
  true for each one. The maintainer deleted "Without the file, placement and
  graph are unchanged" from the #504 CHANGELOG (`dd417bc`); do not bring it
  back.
- Private notes (`xllama-notatki/`, scratchpad) are not source data.
- A data directory follows `bench/results/2026-11-16-series-x-422/README.md`:
  operator-declared and not re-measured by maintainers; console model + OS
  build; package version + MSIX SHA-256 + PFN + CI run/commit (or a named
  gap); method; file inventory; model hashes; caveats (e.g. the selftest CSV
  `host` is hard-coded `xbox-series-s` in `uwp/inference-bridge.cpp`,
  `format_d3d12_selftest_row`); dxc version for generated DXIL. No IP
  addresses or hostnames: run
  `grep -rnE '192\.168|10\.[0-9]+\.[0-9]+\.|\.local\b' <dir>` before commit.
- After every docs edit: `python3 -I .claude/hooks/check_claims.py` (add
  `--file <body.md>` for a PR/comment body). It must print `clean`.
