# cv19 regeneration under the #931 lattice ownership contract

Branch `feat/931-wr90-iris`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XD-wr90iris`.
Pre-change baseline: VESSL run **369367259009** (`rfx-931-base-cv19`), which
reproduced the committed fixture BIT-EXACTLY (0 of 4598 numeric keys differ).

## What was done, and what was already verified without solving

This is the GEOMETRY-PRESERVING redraw: the `+1` on the iris and the `-1` on
the cavity are deleted together with the realization they cancelled, and every
corner moves onto a node plane. Three build-time measurements say the built
filter did not move:

* realized x wall planes at a/90 come back at
  `[[150,158],[214,222],[284,292],[354,362],[418,426]]` — the committed
  pre-change `iris_x_nodes`, plane for plane — with cavities 56/62/62/56 and
  apertures 40/26/24/26/40. Same at a/60 (`t = 5`, cavities 38/41/41/38).
* the declared-vs-realized assert now compares ABSOLUTE plane indices and
  passes at every witness configuration (feed 40 / 70 / 100 cells, b = 4/6/8).
* the FDFD comparator's `self_test` on the new inputs reproduces the committed
  block digit for digit: `empty_s11 4.998689747642886e-14`,
  `unitarity 1.4655321400880439e-09`, `metal_nodes_z 9`, `unknowns 32663`.

So the FDFD leg (three levels, both Richardson estimates, the consistency
witness) is expected to reproduce. The FDTD leg still has to be re-solved,
because one thing did move and is deliberately not compensated back: `span` is
now the true metal extent (276 cells at a/90) where the compensated counts
made it 277, so the trailing feed loses the extra cell it carried and the
domain is one cell shorter, symmetric about the iris stack. The P2 reference
plane moves by one cell into uniform guide.

**The pre-declared falsifier** (design note §5, and the pre-declaration commit
96c55c3a): the realized wall planes must come back at the committed
`iris_x_nodes` (verified above, build-time) and `d_f0` must stay at
+12.08 MHz. A change of order 100 MHz would mean the redraw was not
geometry-preserving. Anything left after that is the one-cell feed change,
which should be small; report the measured `d_f0` against +12.08 verbatim,
pass or fail.

## MEASURED — the pre-declared falsifier, read against the prediction

Run 369367259160 was still in progress when this was written (it had finished
the gated leg and moved on to the coarse rung), so this is the gated result
only; the rest lands with the record.

Predicted, in commit 96c55c3a, BEFORE the run: "the built structure does not
move, so the realized wall planes must come back at today's committed
iris_x_nodes and d_f0 must stay at +12.08 MHz. Anything else means the redraw
was not geometry-preserving and the arithmetic is wrong."

| quantity | pre-change (committed) | post-change (run 369367259160) |
|---|---|---|
| realized iris walls | [[150,158],[214,222],[284,292],[354,362],[418,426]] | identical |
| realized cavities / apertures | 56/62/62/56, 40/26/24/26/40 | identical |
| oracle @ as-realized f0 | 10.95851 GHz | 10.9585 GHz |
| oracle @ as-realized BW | 350.43 MHz | 350 MHz |
| snap df0 vs nominal | +3.3 MHz | +3.3 MHz |
| rfx band | 10.80037-11.14082 GHz | 10.8003-11.1409 GHz |
| rfx f0 | 10.97060 GHz | 10.9706 GHz |
| **d_f0** | **+12.08 MHz** | **+12.12 MHz** |
| d_lo / d_hi / d_bw | +17.08 / +7.09 / -9.99 MHz | +17.0 / +7.2 / -9.8 MHz |
| zeros, span holes, max colpow | 3, 1, 1.0065 | 3, 1, 1.0065 |

**The falsifier passes.** d_f0 moved by 0.04 MHz, which is 0.3% of the residual
and 0.01% of the passband; the filter did not move. That 0.04 MHz is where the
one deliberate non-preserved change shows up: the metal span is now 276 cells
rather than the compensated 277, so the trailing feed lost the extra cell it
carried and P2 sits one cell closer, in uniform guide. It is the size such a
change should be.

What that buys: the case's headline ~12 MHz unexplained rfx-vs-oracle residual
is NOT a lattice-ownership artifact. It survived the contract intact, so the
attribution question the fixture records as open stays open — and it is now
open against a geometry whose realized dimensions are the drawn ones, with no
convention ambiguity left to blame.

## Commands

Two passes, same as cv18 and for the same reason: the `--write-fixture`
self-check demands `GATE_F0_MHZ == round-UP(measured f0 envelope x 1.5)`
exactly, so pass 1 runs with the pre-change constant, prints
`ENVELOPE/GATE MISMATCH (f0)` and exits 1 **by design**; pass 2 runs the
identical command with the re-derived constant and produces the committable
record.

```
# pass 1 and pass 2 use the SAME file
vessl run create -f validation/crossval/_19_iris_filter_results/vessl_931_post_cv19.yaml
```

Local equivalent (do not run on the shared pod):

```
JAX_PLATFORMS=cpu python validation/crossval/19_wr90_iris_filter_aghanim.py --write-fixture
```

Expected runtime: roughly 6 h of FDTD across the nine-configuration witness
population (single gated run 1160 s; manifest `cases[19].cpu_runner.excluded_reason`)
plus the FDFD sweeps at r = 2,3,4 and the 11-point oracle thickness sweep.
Job timeout is set to 32400 s.

| pass | VESSL run id | expected rc | what it is for |
|---|---|---|---|
| 1 | 369367259160 | 1 (gate/envelope mismatch, by design) | measure the new f0 envelope |
| 2 | not yet submitted | 0 | the committable record |

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-cv19-<ts>/`.
The job runs IN the worktree, so `--write-fixture` writes
`validation/crossval/_19_iris_filter_results/rfx.json` and
`tests/fixtures/wr90_iris_filter/fixture.json` directly onto the branch.
**This phase does not commit fixtures**; the ingest phase does.

## Between pass 1 and pass 2 — what to update, and from where

1. `GATE_F0_MHZ` (script line ~219) = `gate_from_envelope(env_f0, quantum=1)`
   from pass 1's printed `envelope <x> MHz -> gate ...` line.
2. `MAX_SPAN_HOLES_GATED` stays 1 unless pass 1 measures more; it is a
   regression lock, so it is raised only with a stated reason, never to pass.
3. The claim_scope's measured digits: `19 MHz`, `12.1230`, `12.0605-12.1230`,
   `d_f0 = +12.08 MHz`, `+17.08 / +7.09`, `-9.99`, `9.80 dB`, `1.0065`,
   `-0.1169` / `-0.1241` cell, the snap figures `+3.3` and `+120.2` MHz, and
   the FDFD deltas `-1.087` / `+13.171` MHz. All of them are pass-1 outputs.
   `test_claim_scope_prose_matches_the_committed_numbers` cross-checks the
   gate and envelope literals against the fixture, so the source edit and the
   regeneration land together.
4. **Decide the edges/bandwidth gating posture** and record the decision. The
   contract removed the half-cell input uncertainty that was the stated reason
   for not gating them; this migration deliberately did NOT gate them, because
   a gate comes from a measured envelope and that envelope only exists after
   pass 1. With pass 1 in hand the repo rule applies directly:
   `gate = round-UP(measured d_lo / d_hi / d_bw envelope x 1.5)` over the
   nine-configuration population. Either gate them at that value or state in
   the claim_scope why not.

## The job, for the record

`**/vessl*.yaml` is gitignored repo-wide, so the submitted file
`vessl_931_post_cv19.yaml` sits UNTRACKED beside this note (and a copy is in
the run's output directory). Everything needed to rebuild it:

| field | value |
|---|---|
| name | `rfx-931-post-cv19` |
| cluster / preset | `remilab-c0` / `gpu-rtx4090` |
| image | `ghcr.io/bk-squared/rfx-openems:5b423bdfe0c8` |
| env | `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=4`, `MPLBACKEND=Agg` |
| mount | `/root/workspace/` <- `volume://remilab-fs/personal-workspaces/` |
| working dir | the worktree itself, `/root/workspace/byungkwan-workspace/research/rfx-931-XD-wr90iris` (NOT a staging copy — `--write-fixture` must land on the branch) |
| pip | `jax[cpu]==0.6.2`, `numpy<2`, `scipy>=1.11`, `h5py>=3.8`, `matplotlib>=3.7`, after uninstalling the CUDA plugins |
| command | `timeout 32400 python -u validation/crossval/19_wr90_iris_filter_aghanim.py --write-fixture` |
| artifacts | `$RUNS/issue931-post-cv19-<ts>/` with `cv19.log`, `cv19.rc`, `produced/`, `produced_files.txt`, `status_before.txt`, `status_after.txt` |

The job exits 0 regardless of the script's rc so the harvest always runs; the
script's own rc is in `cv19.rc`. Modelled on the baseline
`rfx-931-base-cv19` yaml, which reproduced the pre-change fixture bit-exactly.

## Which fixture keys change, and how

* `schema_version` 1 -> 2.
* `gated_rfx` / every row: `aperture_nodes` -> `aperture_wall_nodes` (value
  changes: realized WALL planes, e.g. `[26,64]` -> `[25,65]`), `iris_x_nodes`
  -> `iris_wall_nodes` (value unchanged at a/90).
* `electrical_geometry`: `rule` and `compensation` rewritten (`compensation`
  now begins "none"), new `aperture_cells`, `drawn_aperture_cells`,
  `iris_wall_nodes`, `aperture_wall_nodes`; `drawn_iris_thickness_cells`
  9 -> 8, `drawn_cavity_cells` [55,61,61,55] -> [56,62,62,56];
  `iris_thickness_cells` and `cavity_cells` unchanged (8 and 56/62/62/56 —
  that is the point of the preserving redraw); `cost_of_using_intended_counts_mhz`
  kept as history with a `cost_note` marked HISTORICAL.
* `iris_thickness_zero_count_sweep`: window re-centred from the one-sided
  8.00-8.50 cells to one full cell centred on the realized thickness
  (7.50-8.50, eleven points at 0.1), note rewritten.
* every `s11`/`s21`, all band scalars, the four witness blocks and `gates`:
  re-measured.
* `fdfd_formulation_independent`: expected to reproduce (see above) — if it
  does not, the redraw was not geometry-preserving and the FDTD numbers are
  not to be trusted either.

## Not owned by this group (replacement text is in docs/design_notes/931_migration/)

`validation/crossval/manifest.json` case 19 `claim_scope`, and
`docs/public/guide/benchmarks.mdx` line 67.
