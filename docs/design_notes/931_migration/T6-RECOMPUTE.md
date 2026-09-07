# T6 (oracle / contracts / locks / studio) — recompute ledger for #931

Branch `feat/931-t6-oracle-contracts-locks-studio`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T6-oracle-contracts-locks-studio`.

Everything below is a RE-SOLVE, not a re-tune. Each row says what moved in the
realization, what was pre-declared before the run, and where the run writes.
Nothing in this branch hand-edits a pinned number; the ingest phase re-derives
each one from the run named here.

Submission note: `vessl run create -f` crashes when its cwd is inside a git
WORKTREE (`.git` is a file, not a directory:
`NotADirectoryError: .../.git/HEAD`). Submit from a plain directory with an
absolute `-f` path — the yamls were copied to `/tmp/t6vessl/` and submitted from
there. Source of truth for the yamls is `_vessl931/` in this worktree.

All runs: cluster `remilab-c0`, preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu`,
image `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8`, artifacts under
`/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<ts>/`
(`pytest.log`, `junit.xml`, `commit.txt`; the latest path is echoed to
`issue931-post-<case>.latest`).

Each job copies the LIVE worktree at the moment it starts and writes the commit
it saw into `commit.txt` — read that, not this file, for the provenance of a
given run. The batch was submitted at `b884b83f`; the branch has since merged
`feat/931-lattice-ownership` (core fixes to `rasterize_grid`'s zero-cell refusal
and to `tests/_realized_geometry`'s lane picker) and added two prose commits, so
a job that started after the merge staged the merged tree. None of those changes
touch a declaration, so no row's pre-declaration moves; if a `commit.txt` shows
a commit you do not recognise, `git log` it before reading the numbers.

## Submitted

| case | run id | what moved | pre-declaration | expected runtime |
|---|---|---|---|---|
| `sheet-perturbation-q` | 369367259167 | patch footprint x 13..34 -> 13..35; realized L 5.250 -> 5.500 mm (drawn) | f_mode 24.7530 -> ~23.62 GHz (-4.6 %); Q_f0 roughly unchanged, so the FWHM-ratio tooth survives | 1-2 h |
| `sheet-resonance-ab` | 369367259168 | same patch, three arms | modes 24.5646 -> ~23.44 and 28.1318 -> ~26.85 GHz; the pec-vs-f0 DIFFERENCE unchanged (all arms move together) | 30-60 min |
| `leontovich-alpha` | 369367259169 | plate footprints gained their rim rows (x 380 -> 381 nodes, y 4 -> 5) | alpha UNCHANGED: the extra x node is at the terminated hi-x PEC end outside the fit window, the y rims are PMC faces where the dissipated tangential H is zero | 1-2 h |
| `refplane-thru` | 369367259170 | trace declared a sheet; realized width 4.5 -> 5.0 mm (drawn) | Zc FALLS below the [46.0, 50.5] ohm band and beta/(w/c) rises above [1.03, 1.08]; both bands re-derived from this run against the Phase-0 closed-box flux referee, which is re-run alongside as the independent anchor | 20-40 min |
| `patch-harminv` | 369367259172 | foils declared as sheets — realization measured IDENTICAL to pre-#931 | Leg A -6.17 +/- 1.125 pp and Leg B -6.109 +/- 0.986 pp both HOLD; this is a confirmation run, and a miss falsifies the "identical realization" claim | 40-60 min |
| `patch-s11` | 369367259174 | same migration, Board S | max abs S11 0.9921, in-band min 0.8794, Im(Zin) crossing 8.8189 GHz, in-band max Re(Zin) 4326 ohm all HOLD | 1-3 h |
| `msl-nu-gate` | 369367259178 (and a duplicate 369367259179 — same yaml, submitted twice by a retry loop; either is valid, neither deleted) | NU twin of Board S | same as `patch-s11`, and the two lanes must realize the identical raster (#834) | 1-3 h |
| `conformal-convergence` | 369367259182 | staircase leg now realizes the volume rule (outer face + shorted interior) | staircase `boundary_error` FALLS, so the 1.2x margin may shrink; if BOTH legs collapse toward zero the ring observable has stopped discriminating and needs replacing — a redesign, not a widened margin | 30-90 min |
| `ram-backings` | 369367259193 | `pec_mask_override` backings are index slabs `m[a:b]` — volumes that gain a far face at `b` while the TMM oracle's short sits at the LEADING face `a` | the abs Gamma envelope and both AD-vs-FD legs UNCHANGED on both modules; a move is the far face and turns this row into a re-measure | 20-40 min |
| `farfield-dipole` | 369367259186 | dipole declared a `WireSpec`; edge set verified byte-identical to the old rule on the old mask (Ez 14 edges, Ex/Ey empty) | D stays 2.380 dBi within the +/- 0.5 dBi gate; a move falsifies the byte-identity check | 5-15 min |

`vessl run create -f /tmp/t6vessl/<case>.yaml` for each. Re-submitting a row is
safe — each run stages its own copy of the worktree and writes its own
timestamped output directory — which is why the duplicate `msl-nu-gate` was left
alone rather than deleted (VESSL runs are never deleted here).

## Read back (2026-09-07) — every submitted run against its pre-declaration

The runs below finished; each row is the pre-declaration from the table
above against what the run measured. Three confirmations, four
falsifications, three stale.

| case | run | pre-declared | measured | verdict |
|---|---|---|---|---|
| `conformal-convergence` | 369367259182 | staircase `boundary_error` FALLS; a both-legs collapse would retire the observable | 4 passed, both legs still separated | CONFIRMED |
| `farfield-dipole` | 369367259186 | D stays 2.380 dBi within +/- 0.5 dBi; a move falsifies the byte-identity check | 7 passed | CONFIRMED |
| `ram-backings` | 369367259193 | abs Gamma envelope and both AD-vs-FD legs UNCHANGED on both modules | 20 passed | CONFIRMED |
| `patch-harminv` | 369367259172 | Leg A -6.17 +/- 1.125 pp HOLDS ("a miss falsifies the identical-realization claim") | Leg A **+10.365 %**; Leg B, settling, band and raster all passed | **FALSIFIED — and it named its own cause** |
| `patch-s11` | 369367259174 | crossing 8.8189 GHz, in-band max Re(Zin) 4326 ohm, in-band min abs S11 0.8794 all HOLD | crossing **9.3453 GHz**, in-band max Re(Zin) **0 ohm**, min abs S11 0.9775, Re(Zin) negative across most of the band, Z0 median 79.74 ohm | **FALSIFIED** |
| `msl-nu-gate` | 369367259178 | same as patch-s11 | crossing **9.3407 GHz** — the NU lane reproduces the uniform lane's move to 4 significant figures | **FALSIFIED, consistently** |
| `sheet-perturbation-q` | 369367259167 | f_mode 24.7530 -> ~23.62 GHz (-4.6 %) | f_mode **25.3992 GHz (+2.6 %)** — the wrong way, same family and same sign as the A/B module | **FALSIFIED — cause open** |
| `sheet-resonance-ab` | 369367259168 | modes 24.5646 -> ~23.44 and 28.1318 -> ~26.85 GHz | second mode 30.2153 GHz — the WRONG WAY for a longer patch. **Re-run at HEAD on this pod: identical to 4 significant figures (301 s), so this is NOT staleness** | **FALSIFIED — cause open, see below** |
| `leontovich-alpha` | 369367259169 | alpha UNCHANGED | 3 failed; re-run at HEAD on this pod reproduces the same 3 (o3 field-fit residual 0.0108 vs the 0.01 trust gate) | NOT stale — a real, marginal move; open |
| `refplane-thru` | 369367259170 | Zc falls below [46.0, 50.5], beta rises above [1.03, 1.08]; both bands re-derived from this run | ran at b884b83f, before the refplane helper collected its own sheet: 6 errors + 2 fast-lane failures. At HEAD the fast lane is **27 passed** | STALE — re-submitted |

### What the three board falsifications turned out to be

Not the sheet declaration and not the edge set: the PEC edges are
byte-identical to the pre-#931 rule, which is what the first pass checked.
The MATERIAL under those edges moved. Each board drew its foils as
one-cell PEC Boxes in cells RESERVED for them, with the laminate starting
at the next node; rfx used to re-sample a sheet's own cell material onto
its live edge (#702) so the reserved cell silently became dielectric. The
contract deletes that re-sample, so the reserved cell is vacuum and sits
in series with the cavity. Preflight's own #703 cavity check fired in
every one of those runs and printed the size:

```
Board H  walls 29/34, five cells between, eps_r [1.0, 3.38, 3.38, 3.38, 3.38]
         sum(d/eps) mesh 429.6 um vs physical 232.8 um   (+84.5 %)
Board S  walls 29/34, sum(d/eps) mesh 627.1 um vs physical 429.8 um (+45.9 %)
```

Board H's own docstring names the signature: pre-#702 the isolated patch
read +7.430 % "the ground sheet's own cell assembled as vacuum, diluting
the cavity permittivity". The run measured +10.365 %.

The redraw (commit "the three patch boards put their foils on the laminate
faces") puts each foil ON the laminate face it bounds and snaps the z
origin to the node line. Realized after it, asserted with no solve: walls
28/32, four cells, all eps_r = 3.38, node-to-node 787.000 um = H_SUB
(Board S 788.0 um, the 0.005-cell mesh incommensurability). Preflight's
cavity advisory is silent; the advisory count drops 6 -> 3.

### The two sheet-cavity modules — falsified in DIRECTION, cause open

Both fixtures declare zero-thickness sheets on exact node planes and both
moved UP where a longer patch must move DOWN. What is settled, measured
at build time at HEAD:

```
sheet-resonance-ab, mode "pec": sheet planes {2: [6, 11, 14]}
  plane 11 and plane 14: Ex rows 13..34, Ey cols 15..32
  realized patch node span 5.5000 mm = the declared L_PATCH
```

So the realization is right — the contract's closed footprint gives the
drawn 5.500 mm where the old half-open sampling gave 5.250 mm. A cavity
mode set by that length must fall by the length ratio (24.5646 -> 23.45,
28.1318 -> 26.85, 24.7530 -> 23.63). Both modules moved the other way.

Two candidates, and the module could not tell them apart because it
reported two headline numbers with no trace: the modes really moved up
(which the footprint measurement contradicts), or the two-loudest PEAK
PICKER swapped peaks between arms. This commit adds the R5 census — every
peak with its amplitude, for all three arms — so the next run answers it
by inspection rather than by argument. NOT re-pinned until it does; a
provenance pin re-centred on a number whose mode identity is unknown is
worse than a red gate.

### Re-submitted at the redrawn geometry

| case | run id | pre-declaration (written before submission) | expected runtime |
|---|---|---|---|
| `patch-harminv` | **369367259225** | Leg A moves UP from -6.17 % by roughly the mesh term this module attributes to it (~2 pp), toward its own refinement-ladder plateau near -4.4 %, and STAYS NEGATIVE. A positive value means the slot is still there. Leg B moves < 1 pp (the stub is untouched, and this module measures 0.85 pp per node of stub length) | 40-60 min |
| `patch-s11` | **369367259226** | an Im(Zin) = 0 crossing with Re(Zin) > 500 ohm appears BETWEEN 8.8189 GHz (the pre-#931 pin, on a 983.75 um electrical cavity) and 9.3453 GHz (the un-redrawn run), because the cavity is now 787 um; Re(Zin) stops being negative across the band; Z0 median Re falls from 79.74 ohm toward the Hammerstad-Jensen 50.6 | 1-3 h |
| `msl-nu-gate` | **369367259227** | the same, and the two lanes agree within one DFT bin (#834) | 1-3 h |
| `refplane-thru` | **369367259228** | the 6 physics legs run for the first time at HEAD; Zc and beta/(w/c) are re-derived from this run against the Phase-0 closed-box flux referee | 20-40 min |

| `sheet-resonance-ab` | **369367259230** | the peak census this commit added says whether the modes moved or the two-loudest picker swapped peaks. If the length-scaled 23.45 / 26.85 GHz pair appears in the census, the picker swapped | 30-60 min |
| `sheet-perturbation-q` | **369367259231** | same instrument, same question, on the single-patch sibling (measured 25.3992 against a length-scaled 23.63) | 1-2 h |
| `leontovich-alpha` | **369367259232** | re-measure at HEAD with the full log kept: the pod reproduces the 3 failures, so "alpha unchanged" is falsified and the size of the move has to be read off the run, not argued | 1-2 h |
| `pec-short-sweep` | **369367259233** | the PRE-DECLARED separation: `_vessl931/pec_short_thickness_sweep.py` re-solves at SHORT_CELLS = 1, 2, 4. A total reflector's abs S11 cannot depend on its thickness, so a spread > 0.005 means the REDRAW owns the 0.9670 and the module re-pins from the thickness it declares; a flat sweep means stage C's realized-edge lane owns it and it belongs with the chain-battery re-measure | 20-40 min |

`vessl run create -f /tmp/t6vessl2/<case>.yaml`, submitted from a plain
directory (the worktree crash above still applies). The band constants in
those three lock modules are re-pinned FROM these runs, never from the
arithmetic in this file.


## refplane-thru: the six physics legs are BLOCKED on the preflight group, not on this branch

Re-run at HEAD (VESSL 369367259228): **27 passed, 6 errors**, 13.8 s. The two
fast-lane failures the stale run showed are gone — the refplane helper collects
the sheet it declares now — and the 27 that run are green. The 6 errors are all
one session-scoped fixture, and they are a cross-group dependency:

```
E  refplane thru preflight drifted from the baseline:
   ['_assemble_materials (uniform lane): PEC sheets/wires were classified but
     the caller passed no pec_sheets/pec_wires collector ...']
E  assert ['uncoded'] == ['pec_faces_finite_pec',
                          'wire_port_dead_extent_cells',
                          'wire_port_dead_extent_cells']
```

The fixture asserts the exact advisory CODE list, deliberately ("anything else
= fixture drift, stop"). The trace is a sheet now, and design note §6 says in
as many words that preflight has not been migrated yet — its own
`_assemble_materials(grid)` call passes no collector, so the sheet is invisible
to it and the two conductor-derived codes disappear, replaced by the uncoded
collector warning.

**Not re-derived here, on purpose.** Re-deriving the list now would pin
preflight's un-migrated state and the list would move again the moment the P
group lands `_port_pec_mask`, `_port_transverse_spans` and the wire-port
advisory on `realized_wall_planes`. The ingest phase re-derives it ONCE, after
P. Until then these six are a known, explained red with no number in them.

## Round 2 — the redrawn / re-instrumented runs, read back

| case | run | pre-declared | measured | verdict |
|---|---|---|---|---|
| `patch-harminv` | 369367259225 | Leg A moves UP from -6.17 % toward ~-4.4 % and STAYS NEGATIVE; a positive value means the slot survived. Leg B moves < 1 pp | Leg A **-1.871 %** (negative, slot signature gone); Leg B **-7.077 %**, 0.968 pp of movement, still inside its window; cavity / raster / fidelity assertions all PASSED | falsifier CLEARED; magnitude larger than the ~2 pp guess (see the settling finding) |
| `patch-s11` | 369367259226 | a crossing with Re(Zin) > 500 ohm between 8.8189 and 9.3453 GHz; Re(Zin) no longer negative across the band; Z0 median falls from 79.74 toward 50.6 | crossing **7.7620 GHz** with Re(Zin) peak **4157 ohm**; Re(Zin) positive across the band; Z0 median **60.87 ohm**; preflight cavity advisory SILENT (4 -> 3 advisories) | partly falsified — the crossing moved DOWN, not up, and the reason is physical (see below). Re-pinned from this run |
| `pec-short-sweep` | 369367259233 | abs S11 tracking thickness = the redraw owns it; flat = the operator owns it | min abs S11 1 -> 0.95721, 2 -> 0.96705, 4 -> 0.97607, spread 0.01886 | tracks thickness — and that REFUTES the framing: a total reflector's abs S11 cannot depend on its thickness. Not re-pinned |
| `sheet-resonance-ab` | 369367259230 | the census says whether the modes moved or the picker swapped | the old 28.1318 line is still there at 27.9141 (within one df) at 0.117 amplitude; a 0.418 line at 30.2153 took its place in the top two | PICKER, not the modes. Re-pinned to (25.1741, 30.2153); confirm run 369367259241 **PASSED** |
| `sheet-perturbation-q` | 369367259231 | same instrument, same question | f_mode 25.3992 GHz, reproducing 369367259167 to 4 figures | re-pinned 24.753 -> 25.399; confirm run 369367259242 **PASSED** |
| `refplane-thru` | 369367259228 | the 6 physics legs run for the first time at HEAD | 27 passed, 6 errors — all one preflight-code-list fixture, blocked on the P group | BLOCKED, not re-derived (see above) |
| `leontovich-alpha` | 369367259232 | alpha UNCHANGED | alpha_fit inside its 5 % pin; endpoint-ratio comparator 0.87333 vs 0.72494 (+20.5 %); O3 field fit 0.0108 vs the 0.01 trust gate | partly falsified. Profile dump added and re-run as 369367259243; NOT re-pinned |

### The Board H settling finding

The redraw cleared Leg A's falsifier and exposed a second consequence of the
same change: at `NUM_PERIODS = 120` the UNFED ring-down ends at **-35.43 dB**
of peak against this module's -40 dB truncation bar, while the fed arm still
clears it at -42.21. A 787 um cavity radiates less than the 983.75 um
laminate-plus-vacuum one, so the isolated patch drains more slowly. The bar is
NOT touched; `NUM_PERIODS` goes 120 -> 200, which is what the assertion message
itself prescribes and what the Board S sibling already does (280). Leg A's
-1.871 % is therefore not pinned from the 120-period run — a frequency read off
an unsettled record is what that bar exists to reject. Re-run: 369367259237.

### Why Board S moved DOWN while Board H moved UP

Same redraw, opposite directions, and that is the check rather than a puzzle.
Board H's Leg A is the ISOLATED patch mode: a thinner cavity fringes less, so
it rises. Board S's number is the PORT-PLANE antiresonance of that patch loaded
by a 13.18 mm open feed stub, and a thinner substrate raises eps_eff, so the
stub is electrically longer and its resonance falls. A single spurious global
shift could not move two features in opposite directions, which is why the fed
and unfed terms are pinned separately in the first place.

Confirm runs after the re-pins: `patch-s11` 369367259239, `msl-nu-gate`
369367259240 (re-submitted so it reads Board S's re-pinned band, which it
imports), `patch-harminv` 369367259237.

## Round 3 — the confirm runs

| case | run | result |
|---|---|---|
| `patch-s11` | **369367259239** | **3 passed** (18 m 27 s). Every headline reproduces the evidence run: max abs S11 0.9837, in-band min 0.9096, crossing 7.7620 GHz IN the re-pinned band, in-band max Re(Zin) 4157 ohm, dip 8.800 GHz above the band, preflight advisories 3 with the cavity check silent |
| `msl-nu-gate` | **369367259240** | **2 passed** (19 m 12 s) — the NU twin on Board S's re-pinned band, plus the new build-time parity test (same wall planes, same footprint extents in metres on both lanes) |
| `patch-harminv` | **369367259237** | 7 passed / 1 failed at 200 periods. Settling **-58.30 dB** UNFED and -58.31 FED against the -40 bar (120 periods ended at -35.43), Leg A **-1.886 %**, Leg B **-7.061 %** PASSING. The one red was Leg A against its un-re-pinned window; re-pinned from this run, confirm run **369367259250** |
| `sheet-resonance-ab` | **369367259241** | **1 passed** (24 s) on the re-pinned census pair |
| `sheet-perturbation-q` | **369367259242** | **1 passed, 1 xfailed** (1 m 51 s) on the re-pinned mode-tracking value |
| `leontovich-alpha` | **369367259243 / 369367259244** | envelope regression lock GREEN after the endpoint-ratio re-pin; the two O3 model-trust reds remain, un-widened, and the profile dump explains them (fit residual 0.00245 -> 0.00866, so the two-mode model fits a beatier profile worse) |

The 120-period and 200-period Board H runs put Leg A at -1.871 % and -1.886 %,
0.015 pp apart. The longer record changed the settling, not the physics — which
is the check that the re-pin is a measurement of the board and not of the run
length.

## Fast-lane state of the four T6 directories at the end of this pass

```
tests/contracts + tests/locks   1224 passed, 6 skipped, 18 failed
                                all 18 are tests/contracts/test_example_fidelity_contract.py (13)
                                and test_tutorial_examples.py (5) — the E group's
                                snapshot, regenerated last by plan
tests/studio                     474 passed
tests/oracle                     735 passed, 17 xfailed, 1 failed
                                the 1 is test_pec_short_s11_magnitude, left RED on
                                purpose with the thickness sweep behind it
```

## Measured on this pod, no VESSL needed

**Waveguide chain battery LIVE layer** —
`tests/oracle/test_waveguide_chain_battery_v18_close.py::test_live_cells_reproduce_the_fixture_cpu`,
coarse and mid rungs, run on the shared pod against `fixture_v18_close.json`:

```
[live thru-coarse-false]      max|S_live-S_fixture| = 1.193e-06
[live thru-coarse-flux]       max|S_live-S_fixture| = 2.455e-06
[live pec_short-coarse-false] max|S_live-S_fixture| = 9.381e-01   <-- FAIL vs 1e-4
[live thru-mid-false]         max|S_live-S_fixture| = 1.792e-06
[live thru-mid-flux]          max|S_live-S_fixture| = 1.199e-06
[live pec_short-mid-false]    max|S_live-S_fixture| = 4.979e-01   <-- FAIL vs 1e-4
```

Both rungs, 8 min on CPU. The thru is unchanged at BOTH rungs and the pec_short
fails at both, by different amounts (0.938 coarse, 0.498 mid) — which is what a
phase rotation looks like across two cell sizes, not a broken extraction.

The empty guide is unchanged to 2.5e-6, so the port, the absorber and the
extraction are all where they were. The `pec_short` DUT is the whole delta.
That DUT is a `pec_like` (sigma = 1e10) Box that `_assemble_materials` moves
out of `materials.sigma` into `pec_mask`, and the waveguide S-matrix lane used
to fold that mask BACK into a sigma = 1e10 cell fill for the device run; stage C
(`0184d64c`) replaced the fold with the realized PEC edges. A hard electric wall
is not a 1e10 S/m lossy volume, so a reflection coefficient of magnitude ~1
rotates — 0.938 is the size of that rotation, not a tolerance failure.

Per the inventory's own instruction: **`LIVE_ABS_S_TOL` is NOT widened.** The
fixture family needs a fourth pre-declared measurement run through
`scripts/diagnostics/waveguide_chain_battery_measure.py` (owned by the docs /
scripts group), and the predeclaration in
`docs/design_notes/waveguide_chain_battery_predeclaration.md` needs a §
recording that the device lane changed operator. Handover:
`docs/design_notes/931_migration/T6-waveguide-chain-battery.md`.

**Waveguide port validation battery, PEC short** — measured on this pod after
the §1.5 redraw (the old body was 0.93 of ONE cell against a `dx` the module
never pinned):

```
min |S11| = 0.9670   against this module's 0.99 Meep-class gate
reciprocity advisory 0.0242 vs 0.011 at 7 GHz
```

Left RED, gate not widened. Two changes hit this fixture together — the
geometry (a thicker reflector whose leading face moved up to half a cell) and
the operator (stage C's realized edges replacing the sigma fold on the waveguide
S-matrix lane). Pre-declared separation, in the module docstring and repeated
here: re-run at `SHORT_CELLS = 1` and `4`. If |S11| tracks thickness it is the
geometry and this module re-pins itself; if it does not move it is the operator
and belongs with the chain-battery re-measure. Same class, same handover file.

## Not re-run, and why

* `tests/oracle/test_sheet_film_rta_analytic.py` — measured identical
  realization (x-node 110, 61 Ez / 60 Ey edges). The one-cell-Box and
  zero-extent spellings of `add_thin_conductor` land on the same plane by the
  tie rule, so nothing moved.
* `tests/oracle/test_rcs.py`, `test_oblique_rcs_specular.py`,
  `test_rcs_mie_fixture.py` — sigma fills, fenced out of the contract by §1.8.
  Pinned as a measurement, not a claim, in
  `tests/contracts/test_lattice_ownership_contract.py::test_a_sigma_fill_conductor_is_not_a_pec_body`
  (910 sigma cells vs 912 PEC volume cells on the same sphere).
* Every frozen artifact under `tests/fixtures/` — this group does no numeric
  regeneration (phase-2 scope). Rows whose numbers move are listed in the
  handover files under `docs/design_notes/931_migration/`.
