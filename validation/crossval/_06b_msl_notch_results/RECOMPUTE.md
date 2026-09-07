# cv06b — post-#931 recompute (lattice ownership contract)

Branch `feat/931-crossval-b`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XB-crossval-b`.

## Why every artifact in this directory is stale

The main line and the open stub are now declared as SHEETS (a zero-thickness
Box on the substrate-top node plane) instead of one-cell PEC Boxes, which the
contract would realize as a 63.5 µm filled metal slab with walls at BOTH 254
and 317.5 µm. What that changes on the board, counted build-only (see
`realized_metal()` in the case script):

| quantity | pre-#931 | post-#931 |
|---|---|---|
| PEC volume cells | trace+stub as a node mask | 0 (two sheets) |
| realized wall planes in z | one plane, k=4 | one plane, k=4 (unchanged) |
| geometric realized width (both lines) | 635.0 µm | 571.5 µm |
| electrical width, n_rows·dx (the HJ input) | 635.0 µm | 635.0 µm |
| stub open end | 13652.5 µm | 13589.0 µm |
| quarter-wave length from the line centre | 12350.75 µm | 12287.25 µm (−0.514 %) |

So the solve changes: the stub is one cell shorter. Nothing here can be
translated; it must be re-solved.

## The run

```
vessl run create -f scripts/vessl_931_xb/post-cv06b.yaml      # from a non-git cwd
```

* **VESSL run id 369367259191**, name `rfx-931-post-cv06b`, preset
  `gpu-rtx4090`, cluster `remilab-c0`, image `nvcr.io/nvidia/jax:24.10-py3`.
* Pre-change baseline to compare against: run **369367259002**
  (`rfx-931-base-cv06b`, origin/main d990e18c), outputs under
  `/root/workspace/claude-workspace/rfx/runs/issue931-baseline-cv06b-<ts>/`.
* Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv06b-<ts>/`
  (the pointer file `issue931-post-cv06b.latest` in the runs dir names it).
* Steps and expected runtime, from the baseline: estimator replay (CPU-bound
  replay of a saved sweep, ~2 min), the case script (**~330 s** solve on one
  RTX 4090; a CPU run of this mesh was abandoned at 2 h 52 m), the three build
  falsifiers (**3 × ~330 s**). Whole job ~25–35 min.

`vessl run list` is slow and is not used; read the run by id or read the
`.latest` pointer.

## Fixture keys that change

* `_06b_msl_notch_results/cv06b_build_falsifiers_summary.json` — every value
  under `criterion_A_baseline` (`err_pct`, `bw_ratio`, `witness_bins`,
  `notch_depth_db`, `f_notch_refined_hz`, `f_notch_bin_hz`,
  `f_notch_analytic_hz`, `sub_bin_shift_bins`, `z0_median_ohm`, `solve_s`) and
  under the criterion-B arms (`stub_1cell`, `stub_narrow`);
* `_06b_msl_notch_results/cv06b_falsifier_{baseline,stub_1cell,stub_narrow}.json`
  — full sweeps;
* `_06b_msl_notch_results/cv06b_baseline_run.log` / `.exit`;
* a new `_06b_notch_uniform_logs/<ts>_run.log` (the committed
  `20260827T131217Z_run.log` is the pre-#931 record and stays);
* `tests/fixtures/cv06b_estimator_regate/cv06b_estimator_falsifiers.json`;
* `validation/crossval/manifest.json` `cases[06b].claim_scope` — NOT owned by
  this branch's crossval agent; the replacement text is in
  `docs/design_notes/931_migration/xb-manifest.md`.

## Pre-declared, before the run (design note §5)

Reported verbatim against this list, pass or fail. Full derivation in the
case script's "#931 LATTICE OWNERSHIP" docstring section.

1. **G4 Z0 median stays 46.48 ± 1.0 Ω.** This is the falsifier for the width
   convention: ~46.5 Ω says the electrical width of an n-row strip is n·dx and
   the analytic reference was right to stay at 3.678954 GHz; ~49.4 Ω says the
   geometric span is the electrical width, and then `_realized_trace_width`,
   EPS_EFF, F_NOTCH_AN and G1's shunt-T term all re-derive on 571.5 µm.
2. Measured notch rises ≈0.51 % (the stub's one cell): 3.6255 → 3.644 GHz
   (refined vertex), ±0.3 pp.
3. `err_pct` falls 1.4530 → ≈0.95 %. A RISE above 1.45 % falsifies (2).
4. G2 `bw_ratio` stays 0.9684 ± 0.05 — r = 1 is preserved by construction
   (both lines realize 10 rows), and `assert_realized_metal` now checks it.
5. The half-grid witness (0.3175 bin) and the notch depth (−43.3 dB) are not
   predicted to move.
6. No gate window moves. `NOTCH_FREQ_TOL_PCT` stays 4.0.

## For the ingest phase

The build falsifier summary's `all_ok` is `false` in the committed state
(`stub_1cell` is not sub-bin-visible: 0.145 % measured against 0.532 %
predicted). That is the pre-existing recorded state, not something this
change introduces — do not read a `false` as a #931 regression without
comparing the two `stub_1cell` blocks.
