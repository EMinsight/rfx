# T1 (tests/unit/autodiff, geometry, boundaries) — recompute record

Branch `feat/931-t1-autodiff-geometry-boundaries`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T1-autodiff-geometry-boundaries`.

PI rule, 2026-09-07: the shared pod runs no test lane and no solve longer than
about a minute. Everything below ran on VESSL, one run per case, reading this
worktree from NFS as it stands (commit first — the job does not clone).

The run yamls live in `scripts/` in the worktree but are NOT committed: the
repo's `.gitignore:31` is `**/vessl*.yaml`. Their contents are reproduced at
the end of this file so the ingest phase can recreate them.

## Runs

| run id | what | command | runtime |
|---|---|---|---|
| 369367259135 | the group's three directories at the pre-migration commit 299b0f9f (the failure list this migration worked from) | `vessl run create -f scripts/vessl_931_t1_unit_tests.yaml` | 164 s wall, 12 workers |
| 369367259162 | same lane at 250399d8 (post-migration) | same | ~3 min |
| 369367259163 | per-bin \|S11\| of the cv11-style PEC short, binary and conformal lanes, num_periods 40 and 80 | `vessl run create -f scripts/vessl_931_t1_pec_short_s11.yaml` | ~20 min (4 waveguide S-matrix solves) |

Outputs land in
`/root/workspace/claude-workspace/rfx/runs/issue931-post-t1-*` and
`.../issue931-post-t1-pecshort-*`, with a `.latest` pointer beside them.
`vessl run create` must be issued from a directory that is NOT a git worktree
— the CLI reads `.git/HEAD` as a directory and a linked worktree has a `.git`
FILE, which raises `NotADirectoryError`. Run it from `/tmp` with an absolute
path to the yaml.

## Fixtures NOT regenerated, and why

* `tests/fixtures/msl_replay_accumulators.npz`, `msl_replay_golden_f64.npy`,
  `msl_s_matrix_golden.npy` — the MSL trace migrated to a SHEET with the same
  physical corners (§4 rule 1), which lands on node plane 4 (320 um), the
  plane the pre-#931 rule realized. `test_replay_float64_equivalence` is green
  against the committed goldens (it was red at 2.385e-01 against a 1e-5 gate
  while the trace was a volume). Nothing to recapture.
* `tests/unit/autodiff/test_waveguide_sparam_ad.py`'s three sha256-pinned
  golden S-matrices — they are set by the DOMAIN-FACE PEC convention, which
  §1.8 fences out of the contract. Green, unchanged, no recapture.

## Left for the ingest phase

1. `tests/unit/autodiff/test_forward_outer_jit_traceable.py::
   test_real_interior_pec_under_outer_jit_matches_eager` is RED and must stay
   red until `rfx/geometry/rasterize_grid.py` is patched — see
   `T1-rasterize_grid-outer-jit-tracer-regression.md`. Not T1's file.
2. The five `test_fidelity_topology_findings` reds — three owners, listed in
   that file's module docstring and in `T1-fidelity-sheet-overlap.md` /
   `T1-preflight-junction-plane.md`.
3. `tests/unit/geometry/test_subpixel_pec.py::
   test_pec_short_s11_baseline_unchanged_with_binary_path` — min \|S11\|
   0.9892 against the 0.99 gate. Run 369367259163 is the second witness; read
   its `pec_short_s11.json` before deciding anything. The gate is NOT touched
   here.

## Run yamls (gitignored; reproduce from here)

Both are copies of `scripts/vessl_fast_lane_pytest.yaml` with the checkout
retargeted to this worktree.

`scripts/vessl_931_t1_unit_tests.yaml` differs from the fast lane only in:

    name: rfx-931-post-t1-unit-tests
    env.RFX_CHECKOUT: /root/workspace/byungkwan-workspace/research/rfx-931-T1-autodiff-geometry-boundaries
    env.RFX_LABEL: t1
    pytest target: tests/unit/autodiff tests/unit/geometry tests/unit/boundaries
    OUT: /root/workspace/claude-workspace/rfx/runs/issue931-post-t1-<ts>-<sha>

`scripts/vessl_931_t1_pec_short_s11.yaml` uses the same header and runs, in
place of pytest, a probe that for `conformal in (False, True)` and
`num_periods in (40, 80)` builds
`tests.unit.geometry.test_subpixel_pec._pec_short_sim`, records the realized
x wall planes through `realized_pec_edge_masks` / `realized_wall_planes`, calls
`compute_waveguide_s_matrix(num_periods=..., normalize=False)` and writes each
row as JSON to `pec_short_s11.json`.
