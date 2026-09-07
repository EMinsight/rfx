# #931 Docs group — recompute ledger

The Docs group owns prose, the CHANGELOG, the public pages and
`scripts/diagnostics` migration. Almost none of that is a numeric artifact.
Three things it touched DO need a solve, and this file says which, on what,
and who runs it. Nothing here was hand-edited into a document.

The group has **no pre-change baseline** in
`scratchpad/vessl_baseline/` — there is no `docs.yaml` — so a post-run number
here is a new measurement, not a before/after pair. Where a pre-2.0 number
exists it is quoted in the script's own docstring as dated history and
explicitly not used as a prediction.

## Submitted

### `rfx-931-post-docs-ab` — the sheet-vs-volume declaration A/B

* **yaml**: `scripts/vessl_931_docs_sheet_vs_volume_ab.yaml`
* **preset**: `gpu-rtx4090`, cluster `remilab-c0`, `JAX_PLATFORMS=cpu`
* **reads**: `/root/workspace/byungkwan-workspace/research/rfx-931-Docs-docs`
  (branch `feat/931-docs`), copied to `/root/work/rfx931-docs-ab`
* **writes**: `/root/workspace/claude-workspace/rfx/runs/issue931-post-docs-ab-<ts>/`
* **steps**
  1. `scripts/diagnostics/sheet_vs_volume_patch_radiation_ab.py --check` —
     the build-time realized-plane gate, no solve. It runs FIRST and the job
     stops if it fails; a wrong plane must not be discovered in a Q number.
  2. `scripts/diagnostics/sheet_vs_volume_patch_radiation_ab.py` — two arms,
     120 periods each, `-40 dB` settling witness per arm.
  3. `scripts/diagnostics/patch_sheet_realization_ladder.py --solve` —
     stage 1 (no solve) then three ring-down arms, sheet / vol1 / vol2.
* **expected runtime**: 5 arms of a 29.7 x 18.1 x 12.8 mm patch at
  dx = 196.75 um, 120 periods. The pre-2.0 two-arm version of step 2 ran in
  well under an hour on this preset; budget ~1 h wall, timeouts set to 2 h and
  3 h.
* **pre-declared reading** (both in the scripts' docstrings, committed before
  the run): step 2 passes when `|Q_V/Q_S - 1| <= 0.30` with TM010 present in
  both arms — the declaration does not decide radiation damping at textbook
  scale, which is what the public "foil is a sheet" guidance rests on. A
  narrowing SHEET arm (`Q_S/Q_V > 1.3`) or a missing TM010 in the sheet arm is
  a STOP that qualifies that guidance. A narrowing VOLUME arm is a caveat on
  drawing foil as a one-cell Box, not on sheets. Frequency shift between arms
  is expected and reported, not gated.
* **run id**: `369367259157` (submitted 2026-09-07 10:15 UTC).

## Not submitted — listed with the reason and the owner

### `scripts/diagnostics/patch_tutorial_rfx.py` (migrated, not re-run)

The rfx leg of the canonical patch far-field comparison. Ground and patch are
now sheets at the substrate faces, the rule-2 `zb - dx` compensation is gone,
the graded lane no longer reserves a cell for each foil, and both lanes assert
their realized planes at build time. Its committed result
(`cv05_investigation_results/patch_tutorial_rfx.json`, num_periods=250, wall
4615 s) feeds the **cv05** comparison, which the crossval-A group owns and is
re-solving. Running it here would produce a second, uncoordinated number for
the same comparison.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree> python3 \
        scripts/diagnostics/patch_tutorial_rfx.py

Fixture keys that move: `f_res_ghz`, `spectrum`, `directivity_dbi`,
`hpbw_E_deg`, `hpbw_H_deg`, `meta.z_total_mm` (the fine band lost two cells).
**Owner: crossval-A**, alongside cv05.

### `scripts/diagnostics/msl_probe_clearance_shorted_line.py` (migrated, not re-run)

Ground and trace are sheets, the `-DX` ground compensation is gone, and DX
moved 80 um -> H_SUB/3 = 84.67 um so the 254 um substrate is a whole number of
cells (at 80 um the trace sheet snapped 14 um into the laminate — design note
§1.3, off-lattice interfaces). The recorded 2026-08-28 verdict was measured at
80 um and is a dated result.

    JAX_PLATFORMS=cpu PYTHONPATH=<worktree> python3 \
        scripts/diagnostics/msl_probe_clearance_shorted_line.py

Two arms; the `near` arm was NOT settleable at any run length in 2026-08
(witness -6.1 / -5.7 / -4.7 dB at 150 / 300 / 600 periods), so a re-run must
report the witness before any |S11| number and must not read an unsettled arm.
That is a probe-clearance question, not a #931 question, and re-opening it
belongs with the MSL lane rather than with this contract change.
**Owner: whoever re-opens #726.** Not a #931 blocker.

### `scripts/diagnostics/patch_edgefed_s11_band_repin.py` — `retired` arm is
not re-runnable at all

It monkeypatched `rfx.api._compile.resample_sheet_node_materials`, which #931
deleted. Patching a name a module no longer has is a no-op, so the arm would
have reported `main`'s numbers under the `retired` label. The arm now refuses
with that explanation. The committed
`docs/design_notes/patch_edgefed_s11_band_repin_retired.json` stays as the
#782 falsifier's dated evidence and is not regenerable.

## Artifacts NOT owned by this group

* `tests/data/example_fidelity_snapshot.json` — the examples group regenerates
  it LAST, after preflight and the crossval groups land.
* `validation/crossval/manifest.json`, `validation/crossval/_15_patch_results/`
  — crossval-C.
* `docs/public/gallery/patch_antenna.mdx` — its advisory wording quotes
  preflight text (`port_in_pec`, "the conductor …") that the preflight group is
  rewriting. The gallery assets are a dated record at rfx 1.6.5 / commit
  1eb551b and are NOT re-rendered. Once preflight lands, re-read the advisory
  strings and update the wording only if they changed.
* `docs/public/examples/index.mdx` — every PEC row but `artifact_report_demo`
  is a boundary-PEC control fenced by design note §1.8, so no row edit; the
  `artifact_report_demo` snapshot row is the examples group's.

## SUBMITTED

| run id | name | yaml | submitted | expected wall |
|---|---|---|---|---|
| `369367259157` | `rfx-931-post-docs-ab` | `scripts/vessl_931_docs_sheet_vs_volume_ab.yaml` | 2026-09-07 10:15 UTC | ~1 h (timeouts 2 h + 3 h) |

Submit command (from a directory that is NOT a git worktree — the VESSL CLI
reads `.git` as a directory and a worktree's `.git` is a file):

    cd /tmp && vessl run create -f \
      /root/workspace/byungkwan-workspace/research/rfx-931-Docs-docs/scripts/vessl_931_docs_sheet_vs_volume_ab.yaml

Read the result at
`/root/workspace/claude-workspace/rfx/runs/issue931-post-docs-ab-<ts>/`:
`gate_realized_planes.log` first (it must show one plane per foil in the sheet
arm and two in the volume arm), then `sheet_vs_volume_ab.log` and
`patch_sheet_realization_ladder.log`. Read the settling witness before any Q.
