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
