# T3 — recompute (#931, branch `feat/931-t3-nu-runners-grid-subgrid`)

Machine rule (PI, 2026-09-07): the shared pod runs no solve longer than about
a minute and one small pytest at a time. Everything below goes to VESSL, one
run per case, preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu` inside the job, source
read from this worktree — so the branch is committed before each submission.

No fixture in this group pins a solved NUMBER: the gates are lane parity,
self-comparison digests, loose ratios and one-sided thresholds. What the two
runs below produce is (a) a green/red verdict for the four directories
including the slow marks, and (b) replacements for three MEASURED values that
live in docstrings and are pre-#931. Nothing is hand-edited from either.

## Run 1 — `rfx-931-post-t3-pytest`

* yaml: `docs/design_notes/931_migration/t3_vessl_pytest.yaml`
* command inside the job:
  `python -m pytest -q -n 8 -m "" tests/unit/nonuniform tests/unit/runners tests/unit/grid tests/unit/subgrid`
  with `XLA_FLAGS=--xla_force_host_platform_device_count=2`
* outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-t3-pytest-<ts>/`
  (`pytest.log`, `junit.xml`, `reds.txt`, `pytest.rc`)
* expected runtime: ~35-60 min. The four directories are ~860 tests; the two
  `@pytest.mark.slow` WR-90 iris tests are two-run waveguide S-matrices at
  `num_periods=20` and dominate.
* what a red means: every migrated fixture asserts realized == drawn at BUILD
  time, so a red in one of those build-time gates is a realization defect, not
  a tolerance. Distinguish it from a red in a physics gate before touching
  anything.

## Run 2 — `rfx-931-post-t3-measure`

* yaml: `docs/design_notes/931_migration/t3_vessl_measure.yaml`; producer
  `docs/design_notes/931_migration/t3_remeasure.py`
* command inside the job:
  `python -u docs/design_notes/931_migration/t3_remeasure.py "$OUT/t3_remeasure.json"`
* outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-t3-measure-<ts>/t3_remeasure.json`
* expected runtime: ~40-90 min (dominated by the two iris S-matrices).
* which values it replaces, and why each moves:

  | docstring | pre-#931 value | why it moves |
  |---|---|---|
  | `test_nonuniform_pec_scatterer_limit.py` module docstring: uniform iris | `\|S11\|` ~ 0.6-2.1 | the waveguide S-matrix lane stopped folding `pec_mask` cells into `sigma = 1e10` and applies the realized PEC edges (commit 0184d64c); the fins are also drawn on the node line now |
  | same, NU (graded-dy) iris | `\|S11\|` ~ 1.4-1.6 | same |
  | `test_nu_wire_port_lane_parity.py` module docstring: three-load table, `S11(0.2 GHz)` | `-0.71429+0.00027j` (pec_plates), `-0.60000+0.00034j` (vacuum), "-0.600000 for all three" | the PEC plates realize BOTH drawn faces and short their own interior; before the contract a one-cell body realized one wall and left its normal E live, so the "plates" were films |
  | `test_run_progress_reporting.py::_msl_thru` prose: `Z0`, `beta` | not quoted numerically in the file | the trace moved from a one-cell PEC Box to a sheet on the substrate-top node plane, so it no longer owns cell k=4's material (the #702 backfill is deleted) |

  The gates in all three files are load-independent / ratio / digest gates and
  do not bind on these numbers, so a difference is evidence, not a failure.
  The wire-port row is the one to read carefully: the module's whole argument
  is that the passive reading does NOT move with the load, and the PEC arm is
  now a genuinely conducting body. If the reading starts moving with the load,
  the `#313/#318` normalization story in that file needs re-reading — say so,
  do not adjust the gate.

## Run ids

Submitted 2026-09-07 11:10 UTC from commit 8e223a34, preset `gpu-rtx4090`,
cluster `remilab-c0`, `JAX_PLATFORMS=cpu` inside the job.

| run | id | link |
|---|---|---|
| `rfx-931-post-t3-pytest` | 369367259208 | https://app.vessl.ai/remilab/runs/byungkwan/369367259208 |
| `rfx-931-post-t3-measure` | 369367259209 | https://app.vessl.ai/remilab/runs/byungkwan/369367259209 |
| `rfx-931-post-t3-pytest-r2` | 369367259214 | https://app.vessl.ai/remilab/runs/byungkwan/369367259214 |

Run 369367259208 read commit 6b8f9fae and returned **865 passed, 4 failed,
4 xfailed in 28 min**. All four reds are accounted for:

* `test_distributed_nu_kernel.py` seam pair — the measured lane divergence,
  now `xfail(strict=True)` (see the findings section of the group note);
* `test_auto_config.py::test_auto_mesh_trigger_fires_thin_only_end_to_end` —
  a missing import of the shared helper, fixed in `eadb30b3`, the file's 27
  tests green;
* `test_runner_import_binding.py::test_coax_then_refplane_order_does_not_leak_fake_run`
  — pre-existing slow-lane brittleness, unrelated to #931 (its nested pytest
  passes; the assertion greps the whole stdout for "failed" and a warning
  says "one drive that failed to excite").

`rfx-931-post-t3-pytest-r2` (369367259214) re-runs the same lane on
`cffbb997`, which carries both fixes, so its expected verdict is 2 xfailed
plus that one unrelated red.

Both read this worktree
(`/root/workspace/byungkwan-workspace/research/rfx-931-T3-nu-runners-grid-subgrid`)
as it stands on disk, so a later commit on this branch is NOT in them — check
`commit.txt` in the run's output directory before reading a verdict.
