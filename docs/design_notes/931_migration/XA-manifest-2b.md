# manifest.json — phase 2b replacement text for `05_patch_antenna` (group X-A, #931)

Companion to `XA-manifest.json.md`, which was written before VESSL run
**369367259142** finished and left two slots open: the `realized_stack` sentence
it §3 asks for, and the §4 artifact path. The run has finished and its artifacts
are committed (`efc9d2b1`), so this note fills both. Every number below is read
out of a file that run produced —
`validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json` and
`tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json` (md5
`b9ff9a20520ffcaba19a38e741878abe`) — and nothing is carried across from the old
text by translation.

**Owner of `validation/crossval/manifest.json`: the merge agent.** This note is
the text to apply. `XA-manifest.json.md` §1 (the `claim_scope` opening) was
already applied at `5ce20cd7`; the edits below are what is still outstanding.
Nothing outside `cases[] id="05_patch_antenna"` changes.

---

## A. The four fixture citations in the MODE IDENTIFICATION block

The old citations name `modes[0]` in three places. On the sheet-declared board
the pole ORDER changed, because the board now resolves TM010 in every leg — the
b-axis mode, which the length hook does not touch and which the pre-#931 board
did not resolve at these lengths. `modes[0]` is therefore a correctly identified
TM010 and no longer the mode any of these sentences is about. The index moves to
`modes[1]`; the claims do not change.

| old citation | replacement |
|---|---|
| `runs.baseline.modes[0].freq = 2.331855 GHz`, "-3.76 %" | `runs.baseline.modes[1].freq = 2.446497 GHz`, **+0.95 %** |
| `runs.patch_len_22p0mm.modes[0].freq = 2.993459 GHz` | `runs.patch_len_22p0mm.modes[1].freq = 3.043213 GHz` |
| `runs.patch_len_38p0mm.modes[1].freq = 2.609612 GHz`, "+7.68 %" | `runs.patch_len_38p0mm.modes[1].freq = 2.702632 GHz`, **+11.52 %** |
| `_declared_TM100_hz = 2.423510 GHz` | **unchanged** — closed form in εr / h / L / W, mesh-independent |

The two re-checks `XA-manifest.json.md` §3 demanded, both answered from the
rebuilt fixture and not assumed:

* **Is 22.0 mm still assigned TM110?** Yes. `identify_patch_modes` on the
  rebuilt spectrum returns `[1.9216 GHz → TM010, 3.0432 GHz → TM110]`, refuses
  to name a resonance, and reports "declared DESIGN member TM100 (2.4235 GHz)
  has NO measured mode within 12.50%". Criterion (B) is demonstrated by the same
  length as before; no length was swapped and no tolerance was touched.
* **Does the 38.0 mm falsifier fire again?** Yes, same mechanism. A weak third
  pole (amplitude 3.10e4 against 3.67e5 for its neighbours) lands inside the
  identification window and is named TM100, so that length still PASSES there.
  No amplitude floor was added after the fact, as the existing text refuses.

The sentence "The fixture is regenerated from the repo by
`scripts/diagnostics/build_cv05_ringdown_spectra.py` (VESSL 369367257743, jax
0.6.2 CPU)" gets its run id replaced with **369367259142**.

The block's `rfx_vs_analytic_pct` citation moves to the new record and its value
with it:

> …on the real run (VESSL **369367259142**, 2026-09-07,
> `validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json::openems_mode_id_ok`
> is false while
> `validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json::rfx_mode_id_ok`
> is true,
> `validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json::rfx_vs_analytic_pct = 0.95`
> %)…

## B. The `realized_stack` sentence — §3 of `XA-manifest.json.md`, now measured

Append to the MODE IDENTIFICATION block, verbatim:

> Each run record in the fixture now carries `realized_stack` — the sheet planes,
> the node-to-node cavity and the realized footprints of the build that produced
> it, read from `realized_pec_edge_masks` / `realized_wall_planes` rather than
> restated from the declaration. On the declared build that is: ground sheet on
> node plane k=22 at z = 12.0000 mm, patch sheet on k=28 at z = 13.5000 mm, a
> 1.5000 mm cavity node-to-node over six 250 µm FR4 cells, ground footprint
> 60.0 × 55.0 mm and patch 28.0 × 37.0 mm. The `_realized_x_cell_census` is
> re-measured on every rebuild rather than copied forward, and its unit changed
> with the measurement: it counts the tangential E edges the patch footprint
> stands on — the realized conductor length — instead of the masked nodes, so
> off-lattice rows read one lower than the pre-#931 block (29.5 mm → 28 edges,
> was 29 nodes) while on-lattice rows hold (22.0 mm → 22 either way, 38.0 mm →
> 38). One consequence is recorded rather than hidden: 22.5 mm and 22.0 mm now
> realize the SAME 22 edges, so those two legs are one board with one ring-down,
> and `tests/crossval/test_patch_mode_identification.py::test_cv05_22p5_and_22p0_are_one_realization_since_931`
> fails if a future change separates them again.

## C. `artifact_paths` — §4 of `XA-manifest.json.md`

Add one entry; remove none. The 369367257743 record stays as the before half of
the paired witness (its `openems_*` keys and `declared_modes_hz` are unaffected
by the rfx contract, which is what makes the pair readable).

```
 "artifact_paths": [
  "tests/fixtures/patch_canonical_farfield_e4/patch_farfield_openems.json",
  "validation/crossval/_05_patch_results/cv05_run_openems_369367257743.json",
+ "validation/crossval/_05_patch_results/cv05_run_openems_369367259142.json",
  "scripts/diagnostics/patch_tutorial_openems.py",
  "scripts/diagnostics/patch_tutorial_rfx.py"
 ],
```

## D. What the pair of records now says, for whoever reads the manifest next

Stated here rather than in `claim_scope`, because it is a measurement of the
change and not a claim the case makes:

| | 369367257743 (before) | 369367259142 (after) |
|---|---|---|
| `rfx_harminv_hz` | 2 331 854 551 | 2 446 496 652 |
| `rfx_vs_analytic_pct` | 3.78 | **0.95** |
| `rfx_vs_openems_harminv_pct` | 6.48 | 11.71 |
| `rfx_s11_dip_hz` / `rfx_s11_min_db` | 2.32 GHz / −1.61 dB | 2.48 GHz / −9.62 dB |
| declared members identified | 2 (TM010 missing) | 3 (TM010 −0.76 %, TM100 +0.95 %, TM110 +3.18 %) |
| settling | −21.6 dB | −20.4 dB (both under the −40 dB bar) |

The agreement with openEMS got worse and the agreement with the closed form got
better. That is the direction `rfx-known-issues.md`'s 2026-08-28 A/B verdict
gives for a cavity that became exact, and the `claim_scope` text applied at
`5ce20cd7` already says the pre-#931 openEMS figure is not a baseline this case
preserves. Both arms remain under-settled, so none of these frequencies is
accuracy evidence and the case gates none of them.

## E. Still open after this note

* `_ENVELOPES_REDERIVED_FOR_931` in
  `tests/crossval/test_patch_canonical_farfield_e4.py` and the three constants it
  holds. That file builds its own canonical thirds-rule patch, not cv05's board,
  so run 369367259142 does not measure it. The re-derivation is VESSL run
  **369367259289** (`scripts/vessl_931/cv05_farfield_envelope.yaml`, producer
  `scripts/diagnostics/measure_patch_canonical_farfield_e4.py`). `XA-manifest.json.md`
  §1's rule stands and needs no change either way: the manifest cites
  `D_ABS_TOL_DB`, `F_RES_REL_LO` and `F_RES_REL_HI` by NAME with no band inline,
  so whatever that run measures flows through without a second hand-edit here.
  The pointer `tests/crossval/test_patch_canonical_farfield_e4.py:134,140,141`
  is correct against the file as it stands; whoever flips the flag must re-check
  those three line numbers, because `tests/contracts/test_evidence_citation_pointers.py`
  gates them.
* The cv05 fixture's own reproducibility, VESSL run **369367259288**
  (`build_cv05_ringdown_spectra.py --check` against the committed file at 1e-6
  relative). A MISMATCH there invalidates the citations in §A, so the manifest
  edit should land after that verdict, not before.
