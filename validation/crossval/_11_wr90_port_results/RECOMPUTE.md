# cv11 — post-#931 recompute (lattice ownership contract)

Branch `feat/931-crossval-b`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XB-crossval-b`.

This directory exists only to hold this note: cv11 writes its artifacts to
`tests/fixtures/waveguide_broad_e5/` and to its own stdout, not to a
`_11_*_results/` directory like cv06b and cv07.

## What changed, and why the case must be re-run

1. **The port-aperture trim is deleted.** Both waveguide ports carried
   `y_range=(0, A_WG_REALIZED - DX_M)` / `z_range=(0, B_WG_REALIZED - DX_M)`,
   an in-script correction for the port eigenproblem spanning n_nodes columns
   instead of n_cells. #889 fixed that in rfx
   (`_node_span_to_cell_span`), so the trim became the double correction the
   script's own SEQUENCING paragraph predicted. Measured build-only on this
   checkout:

   | | `cfg.f_cutoff` | vs quote-realized 6.517391 GHz |
   |---|---|---|
   | with the trim | 6.807677 GHz | +4.454 % |
   | without the trim | 6.512162 GHz | −0.080 % |

   The untrimmed value is exactly the 6.512162 GHz the docstring credited to
   the trim in 2026-08.

2. **The PEC short's realized walls are asserted** (`assert_realized_short`,
   no solve): 145.000 / 146.000 / 147.000 mm, the drawn 2 mm plug with walls
   on both faces and the interior shorted. The far face at 147 mm is new
   under the contract; it sits BEHIND the reflector. The reflection plane is
   the first wall, 145.000 mm, unchanged. The cell-relative `2 * DX_M` extent
   is replaced by the absolute `PEC_SHORT_T_M = 0.002`.

3. `A_WG_REALIZED` / `F_CUTOFF_TE10` (QUOTE-REALIZED) are NOT touched: they
   read the realized DOMAIN, whose walls are `BoundarySpec` faces, which
   design note §1.8 fences out of the body-ownership contract.

## The run

```
vessl run create -f scripts/vessl_931_xb/post-cv11.yaml        # from a non-git cwd
```

* **VESSL run id 369367259194**, name `rfx-931-post-cv11`, preset
  `gpu-rtx4090`, cluster `remilab-c0`, image
  `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8`, `JAX_PLATFORMS=cpu` (the
  case pins `JAX_ENABLE_X64=0` for the solver carry and is a CPU case;
  measured 2 m 07 s on a 32-core pod).
* Pre-change baseline: run **369367259004** (`rfx-931-base-cv11`). It exits 1
  — cv11 is a diagnostic reporter and its FAIL against conj(MEEP) on the
  pec-short leg is the known stale state recorded in `manifest.json`
  `cases[11]`. Compare leg by leg, not by exit code.
* Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv11-<ts>/`.
* Expected runtime ~5 min including the pip step.

## Artifacts that change

* the RUN RESULT table in `11_waveguide_port_wr90.py`'s docstring (12 gate
  lines, measured 2026-08-28 WITH the trim — superseded, re-measure);
* `tests/fixtures/waveguide_broad_e5/cv11_wr90_main_baseline_stdout.txt`,
  `cv11_wr90_fresh_stdout.txt`, `cv11_wr90_witness_np400_stdout.txt`;
* `tests/fixtures/waveguide_broad_e5/wr90_rectangular_broad_e4_comparison.json`
  — already flagged stale in `manifest.json` `cases[11]` (0.0186 rebuilt vs
  0.0707 pinned, a 3.7× delta from a June→August code change). The #931
  re-run forces that decision: when it is refreshed, BOTH deltas must be
  explained (June→August code, August→post-#931 realization), never spliced;
* `validation/crossval/manifest.json` `cases[11].claim_scope` — replacement
  text in `docs/design_notes/931_migration/xb-manifest.md`.

## Pre-declared, before the run

1. The pec-short per-bin `|S11|` envelope returns from [0.9980, 1.0019]
   toward [0.9995, 1.0000], and `max_diff` 0.0020 → ~0.0005. Mechanism:
   deleting the trim restores `u_hi == u_grid_size`, which re-enables the
   PEC-ghost aperture-weight zeroing the 2026-04-27 DROP-weight fix depends
   on. An envelope that does NOT tighten falsifies that mechanism.
2. The round-trip phase leg stays near 3.26° max / 1.45° mean: its reference
   cutoff error is unchanged at −0.080 %, and the reflection plane did not
   move.
3. The slab legs (mag 0.0141 / 0.0023, phase 8.94°, complex 0.0654) move only
   if something touched dielectric sampling, which §1.8 says nothing did. A
   test pins that the slab still realizes 11 x-nodes, 95.000 → 105.000 mm.
4. No gate is tightened on the new envelope in this pass — "tighten the
   magnitude gates after #729 settles, on a re-measured envelope" is the
   script's own rule, and #729's port-aperture default is exactly what just
   moved.
