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
* **note on the version the job reads**: submitted before the four migrated
  diagnostics were switched onto the shared
  `tests/_realized_geometry` helper (commit after the merge of
  `feat/931-lattice-ownership`). The change is to the BUILD-TIME gate only —
  it got stricter, three columns per arm instead of one — and to nothing the
  solve touches: geometry, `num_periods`, the settling bar and the pre-declared
  verdict band are byte-identical between the two versions, so the run's arms
  are the arms described here. If the job's own `gate_realized_planes.log`
  shows the one-column form, that is why.

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


---

# RESULT — run 369367259157 (`rfx-931-post-docs-ab`), read 2026-09-07

Artifacts: `/root/workspace/claude-workspace/rfx/runs/issue931-post-docs-ab-20260907T101615Z/`

## Gate 0 — build-time realized planes, no solve: PASS

```
[sheet]  ground 4.1318 mm (node 29) -> walls ['4.1318']
[sheet]  feed   4.9188 mm (node 33) -> walls ['4.9188']
[sheet]  patch  4.9188 mm (node 33) -> walls ['4.9188']
[volume] ground 4.1318 mm (node 29) -> walls ['4.1318', '4.3285']
[volume] feed   4.9188 mm (node 33) -> walls ['4.9188', '5.1155']
[volume] patch  4.9188 mm (node 33) -> walls ['4.9188', '5.1155']
```

One plane per foil declared as a sheet, two per foil declared as a one-cell
volume, each at the declared node. Drawn equals realized in both arms.

## Step 2 — the A/B: **NOT READ**, by its own pre-declared rule

| arm | settling witness (bar −40 dB) | ring-down spectrum |
|---|---|---|
| S (sheet) | **−43.2 dB — SETTLED** | 8.51/Q73/a0.068, 10.87/Q23/a0.025, 12.58/Q20/a0.0056 |
| V (one-cell volume) | **−37.0 dB — NOT SETTLED** | 8.33/Q87/a0.098, 10.75/Q24/a0.037, 12.25/Q71/a0.0023 |

The volume arm failed its witness, so **no Q ratio was computed and none is
claimed**. That is the docstring's rule, applied: "a settling witness must pass
in BOTH arms or the arm's numbers are not read at all."

What may be said, and no more: the two arms are consistent with the volume arm
ringing LONGER (Q87 against Q73 on the TM010, and 3 dB less drained at the same
120 periods), which is the direction the pre-declaration named as "the VOLUME
arm narrows — the 196.75 µm plate closes the cavity". **That is a hypothesis
the run did not test**, because a truncated record inflates apparent Q by the
same sign. The honest next step is a longer record for the volume arm with the
new length pre-declared before it runs — not a re-read of these numbers.

**The public "foil is a sheet" guidance is not contradicted by this run.** The
arm that supports it settled; the arm that did not settle is the one drawing
foil as a plate, which is the drawing the guidance tells you not to use.

## Step 3 — the ladder

Stage 1 (no solve) reproduced the committed table exactly: sheet / vol1 / vol2
all at 787.0 µm gap, `sum(d/eps)` 232.8 µm, **−0.0 %** against the physical
stack. Stage 2 had not written its `.rc` when this was read; re-read
`patch_sheet_realization_ladder.log` in the same directory for its arms.

## MEASURED CONTRADICTION — preflight still describes the deleted rule

Both scripts print preflight verbatim, and on the SAME geometry in the SAME
run preflight says the opposite of what the realization does. This is a
measured finding, not an inference, and it is group **P**'s:

* `'pec' z-extent 196.7µm = 1.0 cells — … A conductor thinner than a cell is
  modelled as a one-cell PEC surface — tangential E is zeroed on it and the
  normal component survives as surface charge … switching to
  add_thin_conductor() would not change it` — **false at 2.0.** A one-cell PEC
  Box is a volume: it shorts its normal edge, and switching to
  `add_thin_conductor()` changes exactly that.
* `2 sheet-bounded cavities differ from the physical stack by more than 1% …
  sum(d/eps) mesh 232.8µm vs physical 174.6µm (+33.3%) … of which 58.21µm is
  geometry[1]'s OWN cell … that sheet fills one cell, and rfx zeroes only
  TANGENTIAL E on a one-cell PEC sheet, so the cell's normal-E edge stays live
  and sits INSIDE the cavity` — the ladder measures the same cavity at
  **−0.0 %** from the realized edge set, and a sheet has **no own cell**. The
  check is comparing node-to-node against face-to-face across a foil thickness
  the sheet model does not have.
* The `[PREFLIGHT] _assemble_materials (uniform lane): PEC sheets/wires were
  classified but the caller passed no collector` line appears as an ADVISORY in
  a run whose sheets are realized correctly — preflight's own assembly is the
  caller that dropped them. Same root as the `first-patch.mdx` strict failure.

Design note §6 already fences these under "Not yet implemented — preflight".
This run is the evidence with numbers attached.
