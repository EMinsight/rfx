# RECOMPUTE — section T (tests/crossval, chain battery, gate policy), #931

Branch `feat/931-t5-crossval-tests`; worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T5-crossval-tests`.

## Submitted by section T

| case | VESSL run id | yaml | command inside the job | expected runtime |
|---|---|---|---|---|
| WR-90 chain battery, post-change re-measure | **369367259196** | `_results_T/rfx-931-post-chain-battery.yaml` (preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu`) | `python scripts/diagnostics/waveguide_chain_battery_measure.py --out-dir $OUT --stages {cells,ad_fd,plane_shift} ... ` then `--stages assemble --fixture-out $OUT/fixture.json` | ~3-6 h wall (run 3 used 4 h + 4 h + 1 h timeouts; three rungs x three DUTs x two lanes, then the AD/FD and plane-shift legs) |

Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-chain-battery-<ts>/`,
with `issue931-post-chain-battery.latest` naming the directory. The job reads
the working tree of this worktree over NFS, so the commit it measures is
whatever is checked out when it starts; `commit.txt` in the output records it.

**Why it must run.** Two things changed for this battery and neither can be
re-pinned by hand:

1. `compute_waveguide_s_matrix` no longer folds `pec_mask` into `sigma = 1e10`;
   it applies the realized PEC edges (`rfx/api/_sparams.py`, "#931 §1.7: the
   interior PEC of this lane is the realized edge set"). So
   `design_override(kind="sigma")` no longer carries the fold — re-stamping it
   would put a lossy volume on top of an already-shorted conductor — and the
   `("pec_short", "sigma")` AD leg now differentiates through a **different
   quantity**. Its gradient is not the old gradient scaled; it is a different
   measurement.
2. The 2-cell PEC short gains a wall at its far face (`x = 63.50 mm`). The
   reflecting near face is unchanged, so `|S11| ~ 1` and
   `pec_short_phase_oracle_deg` should hold — but "should" is the word the
   contract exists to remove, so it is measured.

**Pre-declared, before the run.** The `thru` and `slab` legs must come back
unchanged to within their existing tolerances: neither declares a conductor,
and the absorber depth is unchanged (measured build-time: `guide_source` is
`("domain_faces", "domain_faces")` at every rung, `fc_TE10 = 6.55714 GHz`,
`cpml_layers = 17 / 34 / 68`). `referee_pec_short` must stay inside
`[0.99, 1.03)` with mean within 0.02. The `("pec_short", "sigma")` AD leg is
expected to MOVE and is report-only until re-declared. If the thru or slab legs
move, the diagnosis is wrong and the change is not the PEC short.

The job runs the build-time witnesses (`assert_dut_realizes_its_faces`,
`assert_oracle_anchors_are_realized`) before it spends any solve time, so a
geometry regression fails in seconds rather than in hours.

Fixtures are NOT committed from this phase (the ingest phase does that). The
artifact is written to the run directory only.

## Not submitted by section T — owned elsewhere, named so nothing is lost

| case | owner | what section T is waiting for |
|---|---|---|
| `rfx-931-post-cv19` | crossval-D | `19_wr90_iris_filter_aghanim.py --write-fixture` on the migrated builder. Until it lands, six tests in `test_wr90_iris_filter_gates.py` skip with that name, and `_PINS_REPINNED_FOR_931 = False` holds the four hard pins. |
| `rfx-931-post-cv18` | crossval-D | `18_wr90_iris_modematch.py --write-fixture` with the aperture `- 1` deleted and a `t_c = 1` row added (design note §5 one-cell witness). Two tests skip naming it. |
| `rfx-931-post-cv15` | crossval-C | cv15 re-solved on sheet declarations. All eight tests in `test_crossval_cv15_wall_planes.py` skip until the script imports. |
| `rfx-931-post-cv05` | crossval-A | the canonical patch re-solved on the sheet-declared board. `_ENVELOPES_REDERIVED_FOR_931 = False` holds three slow gates. |
| `rfx-931-post-cv06b` | crossval-B | cv06b re-solved with the trace and stub as sheets. The Z0-anchor test skips naming it; the run-log/live width comparison reds if the log is not refreshed with the declaration. |
| `rfx-931-post-cv07`, cv20 Stage B | crossval-B / E | same foil decision; annotated in the test files. |

## Explicitly NOT re-run (measured, contradicts the tests-crossval inventory)

Five GPU re-runs the inventory asked for are unnecessary. Every one of these
builds its metal with the low-level `rasterize(grid, [(shape, 1.0, sigma)])`
CELL FILL, which design note §1.8 fences out of the ownership contract:

* cv16 ka sweep (`validation/crossval/16_pec_sphere_mie_ka_sweep.py:283`)
* RCS Mie reference (`tests/fixtures/rcs_mie_e4`)
* RCS sphere three-way (`tests/fixtures/rcs_sphere_three_way`)
* RCS cube vs Bempp (`tests/fixtures/rcs_cube_bem/generate.py:58`)
* WR-90 T-junction E4/E5
  (`scripts/diagnostics/build_waveguide_tjunction_broad_e5_envelope.py:41-42, 64-65`)

Also not re-run, with the reason traced to the producer: the WR-90 broad-E4/E5
and NU-E4 `pec_short` legs. The short is drawn two cells thick at `x = 145 mm`
on a 1 mm mesh, so both faces are on node planes; the contract adds a wall at
the FAR face only, behind a total reflector, and the reflecting near face is
where it has always been.
