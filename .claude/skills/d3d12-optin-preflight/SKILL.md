---
name: d3d12-optin-preflight
description: Run before every push or PR of an opt-in d3d12 op or knob (ROPE, bias ADD, next cuts). Walks the checklist the upstream maintainer applied by hand in PR #504 and the PR #511 review, including docs/CHANGELOG/PR-body claims and receipts, and forces a claim-to-test table.
when_to_use: before git push, gh pr create/edit, or replying to a review on a d3d12 branch; again after any commit that touches docs or CHANGELOG
---

# d3d12 opt-in preflight

Local skill (`.claude/` is gitignored; do not add it to a PR). Sources: the
maintainer commits on #504 (`dd417bc`, `364364c`, `e3f79ee`, `6ed22c4`,
including their CHANGELOG/docs rewrites) and the #511 review (2026-11-23).
Evidence rules: `.claude/rules/evidence.md`. The `pre_publish.py` hook denies a
push/PR of a d3d12 branch until this skill stamps the exact HEAD.

## Checklist (each item: evidence, or "not applicable" with a reason)

1. **Opt-in is isolated.** The pipeline and any other resource are created only
   after the knob is read. A failure logs, sets the knob to 0 and leaves the
   rest of D3D12 alive. Read `init_gpu` and check that no new object is built
   with the knob off.
2. **First-use state.** List every lazily created resource the op touches
   (`g.xq`, descriptors, tables). For each, say what happens when the op is
   the first user on a fresh backend, and name the selftest row that runs
   first and the `cold_scratch=1` log line that proves it.
3. **`supports_op` matches what the dispatch can bind.** Every buffer the
   kernel binds as a UAV must be one that has the UAV flag (`D3D12_Host`, not
   `D3D12_Weights`). Every refusal path has a rule in `d3d12_*_supported` and a
   case in `tests/test_ggml_d3d12.cpp`.
4. **Selftest row + gate.** A `d3d12be.flag` row per kernel, required by
   `scripts/bench-d3d12-selftest.sh` (`--require-<knob>`), with a test of the
   gate in `tests/test_bench_run_status.py`.
5. **Evidence gate fails closed and is machine-checked.** `--profile` in
   `scripts/paired-bench-summary.py`: exactly N complete alternating AB/BA
   pairs, measured-phase counters only (no warm-up fallback), mixed config
   rejected, tests for full / incomplete / mixed / missing counter.
6. **Docs say what the code does.** Re-read each sentence of the doc section,
   CHANGELOG entry and PR body against the code: what is keyed, summed,
   refused. The gate's contract and the docs' wording match (NMSE bound is
   not "bit for bit").
7. **Source data.** `python3 -I .claude/hooks/check_claims.py` prints `clean`,
   and `--file <body.md>` is clean for the PR body. Figures stay out of docs
   unless a `bench/results/<dir>` lands in the same PR.
8. **Failure paths.** Table: each fallback/refusal sentence → every failure
   path → code line. A sentence true for one path only is rewritten or cut.
9. **Assert vs refusal.** `git diff origin/main..HEAD -U0 | grep -n GGML_ASSERT`.
   Every new assert reachable with the knob on becomes a refusal in
   `dev_supports_op` plus a case in `tests/test_ggml_d3d12.cpp`. Not "mention
   it in the PR".
10. **Artifact provenance.** dxc version for every new
    `shaders/generated/*_dxil.h` is in docs or the PR body; labels such as the
    selftest CSV `host` (hard-coded `xbox-series-s`) are explained.
11. **CHANGELOG.** No sentence the maintainer already deleted ("Without the
    file, placement and graph are unchanged"); parity narrowed to where the
    data lives ("comparisons reported in #N").

## Bots and reviews

After a push, read the **review body and the outside-diff section**, not only
inline comments: `gh api repos/<owner>/<repo>/pulls/<N>/reviews`. CodeRabbit
puts its most useful findings under "Outside diff range comments". Write each
point as fixed / rejected with a reason, all in one commit.

## No claim without a test

Every thesis in the PR text ("without the file nothing changes", "bit-identical
logits", "cold start is safe") needs a row `claim -> test that checks it`. If
there is no test, either write one or drop the claim. Say "not measured on
Series S" explicitly when it is Series X only.

## Before the push

- `ctest`, `clang-format` 22.1.5, `prettier` 3.9.6, `shellcheck` 0.11.0,
  `python3 scripts/check-coherence.py`, the unittests touching your files.
- `scripts/check-win-syntax.sh` if an xwin splat exists; otherwise state in the
  PR that Windows code compiles only on `build-uwp`.
- An independent `/code-review high origin/main..HEAD`; do not hand it your
  conclusions.

## Stamp

Only when every item above has evidence: `bash .claude/hooks/preflight_stamp.sh`.
Any later commit invalidates the stamp; run this skill again.
