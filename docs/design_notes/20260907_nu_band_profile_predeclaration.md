# Axis-agnostic band profile builder (`make_band_profile`) — pre-declaration

**Status:** pre-declaration. This commit precedes any fix, any builder code
and any fix-validation measurement. Every numeric window below is frozen
from this commit onward; results are appended under "Results", never
edited into the windows.
**Class:** declared-vs-realized, mesh builder edition (family #325, #763;
siblings #740, #745, #752, #811).
**Tree:** worktree `rfx-nu-band`, branch `feat/nu-band-profile`, based on
`origin/main d990e18c` (`d990e18ce0870ac7a893a86627e7232d1cf92c7b`).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-band/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Date: 2026-09-07 (KST).

## 1. Defects, re-reproduced on this tree (zero FDTD)

The lead measured all three on `main d990e18c`. Each was re-run here with
the tree pinned; numbers match unless a discrepancy is stated.

### Defect 1 — `rfx/auto_config.py::_make_dz_profile` on a 5-layer PCB stack

Fixture: layers from z = 0.5 mm upward: core 0.8, prepreg 0.1, core 0.8,
prepreg 0.1, core 0.8 mm, all eps_r 4.3; `domain_z` 4.0 mm; `dx` 0.2 mm
(`_make_dz_profile(feats, 4.0e-3, 0.2e-3)`).

Measured here: nz = 45, sum = 4.000000 mm (exact), dz_min = 8.333 um,
max adjacent ratio **8.000** at index 30 (8.333 um next to 66.667 um), 25
adjacent ratios > 1.4, every declared interface on a node (max error
8.7e-19 m). Realized cells (um):

```
[136.0, 104.6, 90.7, 69.7, 53.6, 45.3 | 66.7, 133.3, 200, 200, 133.3, 66.7 |
 8.33, 16.67, 25, 25, 16.67, 8.33 | 53.3, 106.7, 160, 160, 160, 106.7, 53.3 |
 8.33, 16.67, 25, 25, 16.67, 8.33 | 66.7, 133.3, 200, 200, 133.3, 66.7 |
 51.2, 66.5, 86.5, 102.3, 133.0, 153.5, 153.5, 153.5]
```

Matches the lead (nz 45, 8.333 um, ratio 8.000 at idx 30, 25 ratios > 1.4).
Cause, confirmed by reading the code: each feature block is realized
uniform (`n = max(4, ceil(thickness/dx))`), `apply_thirds_rule` then splits
the cells on BOTH sides of every boundary index into 2/3 + 1/3, and
`_smooth_preserving_blocks` smooths only the free (air) runs — a seam
between two adjacent protected blocks (core | prepreg: 200/3 = 66.7 um next
to 25/3 = 8.33 um) can never receive a ramp.

Two facts the reading adds to the lead's statement:

- The thirds rule fires at **every** feature boundary index, including a
  dielectric | dielectric seam (core | prepreg) where its stated rationale
  ("1/3 inside the conductor, 2/3 outside") has no conductor. This is where
  the 8.33 um cells — the dz_min of the whole column — come from. Ownership
  question, #931, out of scope here (section 5).
- The middle core (1.4 to 2.2 mm) realizes **5 cells of 160 um** while the
  two identical 0.8 mm cores realize 4 cells of 200 um: `2.2e-3 - 1.4e-3`
  evaluates to `8.000000000000002e-4`, so `ceil(span/dx)` returns 5. A
  floating-point ceiling on a physically exact quotient. Design refinement
  R3 below.

### Defect 2 — `rfx/nonuniform.py::make_z_profile` has no descending ramp

Fixture: `make_z_profile([1.0e-3, 1.2e-3, 2.5e-3, 2.7e-3], 4.0e-3,
50e-6, 200e-6, 1.4)`.

Measured here: nz = 32, sum = 4.000000 mm, all five feature coordinates on
a node (max error 4.3e-19 m), but three adjacent ratios above the cap:
**4.527** at index 6 (226.36 -> 50 um), 3.764 at index 19 (188.18 -> 50 um),
1.600 at index 22 (80 -> 50 um); last cell 188.18 um, not the 50 um
boundary cell the docstring ("fine -> coarse -> fine") implies. Cells (um):

```
[50, 70, 98, 137.2, 192.1, 226.4, 226.4 | 50 x4 | 50, 70, 98, 137.2, 192.1,
 188.2 x4 | 50, 70, 80 | 50, 70, 98, 137.2, 192.1, 188.2 x4]
```

Matches the lead (ratio 4.53, 226.4 -> 50 um, last cell 188 um). Cause:
the per-segment loop ramps up from the left edge only and fills the rest
with coarse cells; nothing ramps back down before the next feature. Only
caller: `docs/public/guide/nonuniform-mesh.mdx` (lines 97-113), which then
tells the reader to run `smooth_grading` on the result — Defect 3.

### Defect 3 — public `smooth_grading` shifts every downstream interface

Fixture: `cells = [0.2e-3]*5 + [0.05e-3]*4 + [0.2e-3]*5` (a 0.2 mm slab
at z in [1.0, 1.2] mm on four 50 um cells), `smooth_grading(cells,
max_ratio=1.3)`, then `Simulation(domain=(3e-3, 3e-3, 0.0), dx=0.2e-3,
dz_profile=sm, cpml_layers=4, freq_max=10e9, boundary="cpml")` with the
slab as a `Box((1e-3, 1e-3, 1.0e-3), (2e-3, 2e-3, 1.2e-3))` in fr4.

Measured here: 14 -> 24 cells, column 2.2000 -> 3.2749 mm, slab-top edge
missed by 46.2 um (nearest realized edge 1.1538 mm); preflight
`graded_box_rasterization`: "Box material 'fr4' rasterizes to **2 z cells
(implied 4.0)** over z-span [1mm, 1.2mm)". The same signature appears at
max_ratio 1.4 (column 3.0671 mm, edge missed by 44.9 um).

**Discrepancy with the lead's numbers:** the lead quotes "1 z cells
(implied 3.0)". A sweep over slab position (2-5 coarse cells below), fine
count (3, 4) and ratio (1.3, 1.4) on this tree never produced that pair; the
3-cell variants stay silent (2 nodes >= ceil(0.5 x 3)), the 4-cell variants
report "2 (implied 4.0)". Same defect class and mechanism (transition cells
inserted, downstream interfaces shifted, #325 advisory reproduced); the
lead's exact fixture was not recovered from the numbers given. Recorded, not
resolved.

## 2. The design (the lead's decision, with refinements where the code read contradicts it)

New public builder in `rfx/nonuniform.py`, exported from `rfx/__init__.py`:

```
make_band_profile(edges, cell_sizes, *, max_ratio=1.4, protected=None,
                  boundary_cell=None, min_cells=1) -> np.ndarray  (float64 cells)
```

- `edges`: sorted physical coordinates of every interface that must land
  on a node plane, including 0 and the domain end. `cell_sizes`: target
  cell size per segment (`len(edges) - 1`). `protected`: per-segment bool
  (default False). A protected segment is realized UNIFORM
  (`n = max(min_cells, ceil(span/target))` equal cells, no ramp cells
  inside); a free segment may host ramp cells.
- Invariants the output MUST satisfy for ANY valid input:
  - **I1** every edge is a cumulative node coordinate, |err| <= 1e-12 m;
  - **I2** every adjacent ratio <= max_ratio (+1e-9), INCLUDING seams
    between two adjacent protected segments;
  - **I3** sum == edges[-1] - edges[0] to 1e-12 m;
  - **I4** with `boundary_cell` given, `cells[0] == cells[-1] ==
    boundary_cell` exactly (the x/y CPML contract,
    `rfx/nonuniform.py:355-370`: `dx_profile[0] == dx_profile[-1] == dx`;
    dy ends must equal each other), achieved by ramping inside the end
    segments;
  - **I5** axis-agnostic: the same function serves `dx_profile`,
    `dy_profile` and `dz_profile`.
- Ramps are geometric with ratio <= max_ratio, placed INSIDE the coarser
  free segment on whichever side needs it (ascending and descending), and
  the free segment is renormalized to its exact declared length. Two
  adjacent PROTECTED segments with a seam ratio > cap: the coarser
  protected segment's uniform cell size is reduced (more cells) until the
  seam ratio <= cap — interfaces stay exact, the ratio law holds, the cost
  is cells, not accuracy. Iterate to a fixed point (a reduction can create
  a new violation at its other seam).
- Rewire `make_z_profile` on top of it (same signature; the docstring's
  "fine -> coarse -> fine" becomes true) and rewire `_make_dz_profile`'s
  smoothing step so the auto z mesh satisfies I1-I3 (thirds rule kept
  as-is, before the smoothing step, exactly like today). The #763 preserve
  tests keep passing bit-identically for the demo fixture; if the generic
  fixture's realized cells change, justify by the invariants and re-pin
  with provenance.
- Preflight/docs: replace the "up to 3 fine bands" wording in
  `docs/guides/support_matrix.md` and
  `docs/public/guide/nonuniform-mesh.mdx` with the transition law
  (per-transition reflection set by ratio r, local cells/lambda and band
  width; band count only sums) plus the new unwitnessed-range statement
  (bands narrower than the measured width, in-plane still uncovered). Keep
  every existing number; add none that this lane did not measure.
  CHANGELOG entry under Unreleased.
- Tests: `tests/unit/nonuniform/test_band_profile_builder.py` — the PCB
  fixture, the `make_z_profile` fixture, seeded random-stack fuzz, a dx/dy
  round trip, and the revert-proof test (pins the NEW numbers; the OLD ones,
  8.000 / 4.527, quoted in its docstring).

### Refinements (each one is a place where the code read contradicts the design as written)

- **R1 — I2 and the thirds rule cannot both hold inside a block.**
  `apply_thirds_rule` splits a boundary cell d into [2/3 d, 1/3 d]; the
  ratio between those two sub-cells is exactly **2.0**, and between the
  full cell and the 2/3 sub-cell exactly **1.5**, by construction. #763
  already accepted this (its air-run-only ratio re-pin, commit 6973459).
  So, for the auto z path (`_make_dz_profile`), I2 is evaluated on every
  adjacent pair EXCEPT the pairs internal to a thirds split, and the
  protected-protected seam rule compares the ACTUAL seam cells, i.e. the
  1/3 sub-cells (PCB: 66.7 vs 8.33 um today). The public builder itself
  never sees a thirds split, so I2 holds globally there. Consequence for
  the PCB fixture, derived by hand from the rule: prepreg 4 x 25 um
  (thirds 8.333 um) forces `d_core/3 <= 1.4 x 8.333 um`, i.e.
  d_core <= 35.0 um -> **23 cells of 34.783 um per core**, seam ratio
  11.594/8.333 = **1.391**; reference realization nz = **105** (today 45),
  dz_min **8.333 um unchanged**. Cells, not accuracy — and 2.3x the cells.
  That cost is the #931 question, not a reason to widen the cap.
- **R2 — renormalization must never rescale UP and must not touch a pinned
  boundary cell.** `_smooth_preserving_blocks` states "f <= 1 by
  construction since smoothing only inserts". With ramps placed by the new
  builder that no longer holds: a geometric ramp shorter than its segment
  with no room for a full plateau cell leaves f > 1, and an up-rescale
  raises the seam cell above cap x neighbour (checked by hand on the PCB
  top air run: 8-cell ramp 558.3 um in a 900 um run, remainder 341.7 um;
  a single 341.7 um plateau cell gives ratio 1.997). Rule: plateau cells
  are `rem / ceil(rem / target)` (never above target; the ramp's top cell
  is >= target/cap, so the plateau seam stays <= cap); when the ramps alone
  exceed the segment, uniform DOWN-scale f < 1 of the un-pinned cells (the
  #763 method: internal ratios unchanged, seam ratios only improve);
  `cells[0]`/`cells[-1]` pinned by `boundary_cell` are excluded from any
  rescale.
- **R3 — cell count from a float quotient.** `n = max(min_cells,
  ceil(span/target - 1e-9))` (relative tolerance on the quotient), so
  0.8 mm / 0.2 mm realizes 4 cells for every 0.8 mm core, not 4-5-4 as
  today (Defect 1 measurement).
- **R4 — `boundary_cell` preconditions.** An end segment carrying a pinned
  boundary cell must be free (a protected end segment whose uniform cell is
  not `boundary_cell` is a contradiction -> `ValueError`), and long enough
  to hold the pinned cell plus its ramp (`ValueError` otherwise, message
  naming the segment and the minimum span). The fuzz family only generates
  feasible inputs (end segments free, span >= 4 x boundary_cell, with
  boundary_cell equal to the end segment's target).
- **R5 — thin free segments are refinement sources too.** A free segment
  too thin to host any ramp (say 20 um between two 3 mm protected layers
  at 150 um) realizes one 20 um cell; I2 then requires the neighbours to
  come down: a protected neighbour is refined by the seam rule, a free one
  ramps. The fixed-point iteration therefore runs over ALL seams, not only
  protected-protected ones. The fuzz family (thicknesses 20 um-3 mm,
  targets 0.05-2x the layer) exercises this; the resulting cell counts are
  REPORTED (max nz over the family), not gated.
- **R6 — the in-plane cap is 1.3, not 1.4.** `_PreflightMixin._INPLANE_RATIO_CAP
  == 1.3` and `test_inplane_grading_lock_stays_at_1_3` lock it (WP6R.8: no
  in-plane witness exists). The builder's default 1.4 is fine for z; the
  F5 dx/dy round trip must call it with `max_ratio=1.3`, and the auto z
  path keeps its current 1.3 (the #763 air-run pin `<= 1.301`). Neither
  is a cap move.
- **R7 — absorber runway (open decision for the lead).** `boundary_cell`
  pins ONE cell per end; preflight's `nu_grading_reaches_absorber`
  (`rfx/api/_preflight.py`) wants `cpml_layers` uniform interior cells
  against each absorbing face, ratio deviation <= 1e-6. Under R2 the
  plateau next to a pinned cell is `rem/ceil(rem/target)`, generally NOT
  equal to `boundary_cell` (F5 fixture: 12 mm end segment, ramp 0.65 +
  0.845 mm, remainder 10.505 mm -> ten cells of 0.9505 mm beside the 1.0 mm
  pin, ratio 1.052), so that advisory WILL fire on a pinned profile unless
  the builder also holds a runway of `cpml_layers` cells at exactly
  `boundary_cell` and absorbs the remainder in the ramp (re-solving the
  ramp ratio rho <= cap; feasible here with a 10-cell runway and a 3-cell
  ramp at rho = 1.23). Not required by I1-I5; F5 therefore REPORTS this
  advisory and the realized end-run cells, and gates on the invariants
  only. If the lead wants the runway inside the builder, it is a
  `boundary_cells: int` argument added BEFORE F5 is measured, with F5's
  report line promoted to a gate in the same commit — not after.
- **R8 — `_make_dz_profile` dispatch.** `min_cells_per_feature=4` maps to
  `min_cells=4` on protected segments; free (air) segments keep `min_cells=1`
  (today's `max(1, round(gap/dx))`). The pre-existing "gap or top air
  <= dx/2 is dropped" limit (#763 note) is untouched by this lane.

## 3. Falsifiers (frozen; tolerances never widened after measurement)

Fixtures named here: (a) PCB 5-layer (Defect 1); (b) `make_z_profile`
(Defect 2); (c) #763 demo (h_sub 254 um, dx 190.5 um, column 1.754 mm)
and generic two-layer (layers 0.2-0.5 and 1.1-1.35 mm, column 3.0 mm,
dx 0.3 mm); (d) seeded fuzz, `np.random.default_rng(20260907)`, >= 200
stacks: 1-8 layers, thicknesses 20 um-3 mm, per-segment targets 0.05-2x
the segment, protected mix, boundary_cell on/off (R4 feasibility).

- **F1 (I1, interface snap):** every declared edge within **1e-12 m** of a
  cumulative node, on (a), (b), (c), (d), and on the auto-z output of
  `_make_dz_profile` for (a) and (c). Today: (a) and (b) already hold
  (8.7e-19, 4.3e-19 m); Defect 3's public path misses by 46.2 um.
- **F2 (I2, ratio law):** every adjacent ratio <= max_ratio + **1e-9** on
  (a) via the builder, (b), (d); on the auto-z (a) and (c) outputs, every
  pair outside a thirds split (R1) <= 1.3 + 1e-9, and every
  protected-protected seam (actual seam cells) <= 1.3 + 1e-9. Today: (a)
  8.000, (b) 4.527 — these two numbers are the revert-proof.
- **F3 (I3, column):** |sum - (edges[-1] - edges[0])| <= **1e-12 m** on all
  fixtures.
- **F4 (I4, boundary pin):** with `boundary_cell` set, `cells[0] ==
  cells[-1] == boundary_cell` bit-exactly (`==`, no tolerance) on (d) and
  on the F5 profiles.
- **(c) locks, verbatim from `test_auto_dz_profile_preserve.py`:** demo
  substrate-top edge <= 1e-12 m; post-thirds block
  `[63.5, 63.5, 63.5, 42.333, 21.167] um` bit-identical (`np.array_equal`);
  dz_min 21.167 um; column 1.754 mm to 1e-12 m; air-run ratio <= 1.301;
  generic fixture: four interfaces on nodes, both blocks bit-identical. No
  protected-protected seam exists in either, so the seam rule never fires
  there; the air runs MAY re-realize (geometric ramp from the seam instead
  of smooth_grading insertion + plateau drop). If they do, the old and new
  air cells are printed side by side in the results and the change is
  accepted only because the four locks above still hold — nz is not a lock.
- **F5 (I5, axis round trip):** `dx_profile = dy_profile =
  make_band_profile([0, 12e-3, 15e-3, 27e-3], [1e-3, 0.5e-3, 1e-3],
  max_ratio=1.3, boundary_cell=1e-3)`, `Simulation(freq_max=10e9,
  domain=(sum, sum, 10e-3), dx=1e-3, dx_profile=..., dy_profile=...,
  boundary="cpml", cpml_layers=8)` constructs with **zero** warnings
  containing "adjacent cell ratio", `preflight()` does not emit
  `nu_grading_ratio_beyond_validated_cap`, I4 holds bit-exactly on both
  profiles, and the realized interior extents from the grid's own node
  coordinates (`make_nonuniform_grid(...)` x/y cumulative sums, CPML pad
  excluded) equal 27 mm to **1e-12 m** on both axes. Same profile as
  `dz_profile`: clean as well. `nu_grading_reaches_absorber` and the
  realized end-run cells are REPORTED, not gated (R7).
- **F6 (locked-value audit rule, as in the #763 note):** every committed
  value that moves is listed with old -> new and accepted ONLY if the
  realized mesh now satisfies I1-I3 (and I2 under R1) and the moved value
  follows from that by arithmetic; re-pinned WITH provenance in the commit
  message. A moved value that cannot be justified => STOP and report.
  Batteries (declared):
  ```
  pytest tests/unit/nonuniform/test_band_profile_builder.py tests/unit/nonuniform/test_auto_dz_profile_preserve.py \
         tests/unit/nonuniform/test_smooth_grading_preserve.py tests/unit/grid/test_auto_config.py -q
  pytest tests/unit/nonuniform tests/unit/grid tests/unit/preflight -k "nonuniform or nu or profile or auto_config or mesh_planner" -q -o addopts="" -m "not gpu"
  pytest tests/unit/nonuniform/test_multiband_nu_envelope.py -q -o addopts="" -m "not gpu and not slow_physics"
  pytest tests/contracts -q -o addopts="" -m "not gpu"
  ruff check rfx/ tests/ --select E,F,W --ignore E501,F401,E741,E731,E701,E702,E402
  ```
  Bucket (a) candidates read on this tree: `tests/unit/grid/test_auto_config.py`
  (thirds tests 223-264 use `apply_thirds_rule`/`_make_dz_profile` directly;
  `test_make_dz_profile_applies_thirds_rule` pins free-run ratio <= 1.3),
  mesh_planner / auto_configure consumers (nz, dz_min, dt, memory
  estimates), `tests/_example_fidelity_lib.py` (explicit `dz_profile`
  vectors from `fixtures.py`, builder not called — expected no move),
  preflight NU tests (explicit profiles — expected no move).

### F7 — chain model on the OLD vs NEW PCB profile (no FDTD)

Instrument: `validation/research/multiband_nu/chain_model.py::scattering`
on the profile embedded in W2-count runways — **140 lead cells of the
profile's own first cell, 150 tail cells of its last cell** (the chain
solve needs uniform ends equal to the profile ends; the W2 1 mm runway
cells cannot be used without adding a 0.136 -> 1 mm transition that is not
the builder's). F0 = 10 GHz, transverse dy = dx = 0.2 mm (the fixture's
own dx), b = B_Y = 30 mm (TE10, fc 5 GHz), dt from
`make_nonuniform_grid` on the embedded profile: **2.7471e-14 s** (set by the
8.333 um cell, common to OLD and NEW). "Step" = each adjacent pair with
ratio != 1; its single-step value = `scattering([d_k]*140 + [d_{k+1}]*150)`.

Reference numbers (hand realization of the design per R1-R3; the
implementer's builder must reproduce the per-layer cell counts
23 / 4 / 23 / 4 / 23 (core / prepreg / core / prepreg / core, before the
thirds splits) or write down why):

| profile | nz | total \|R\| | total \|R\|² | Σ\|R_step\| | (Σ\|R_step\|)² | Σ\|R_step\|² | max non-thirds step |
|---|---|---|---|---|---|---|---|
| OLD (this tree) | 45 | 9.756e-6 | 9.518e-11 | 5.065e-4 | 2.566e-7 | 1.481e-8 | 1.556e-5 (136.0 -> 104.6 um, r 1.300) |
| NEW (reference) | 105 | 5.747e-5 | 3.303e-9 | 1.267e-4 | 1.606e-8 | 1.958e-9 | 2.955e-5 (122.2 -> 171.1 um, r 1.400) |

OLD's ratio-8 seams reflect **9.0e-6** each (indices 11 and 30): at 8-67 um
cells a 10 GHz wave sees lambda/450 and worse, so the defect is invisible
to a reflection number at this frequency. It is a ratio-law / dz_min defect,
and **F2 is the gate for it, not F7.** F7 is a consistency witness on the
NEW profile:

- **F7a (frozen):** `total |R(P)|^2 <= 1.5 x (Σ_steps |R_step|)^2` on the
  builder's actual output P. Reference window **2.409e-8**, reference total
  **3.303e-9**. This is the lead's rule with the sum taken in AMPLITUDE
  (coherent), not power. Reason, from the chain model itself: the W2
  r = 1.4 two-step ramp (two steps of 1.1832) reflects **1.9536e-3**, while
  one 1.1832 step reflects **8.307e-4** — steps closer than a wavelength add
  in amplitude (2 x 8.307e-4 = 1.661e-3 is the right order; the incoherent
  sqrt(2) x 8.307e-4 = 1.175e-3 is not). On the reference NEW profile the
  literal power rule, 1.5 x Σ|R_step|² = **2.937e-9**, is BELOW the total
  3.303e-9 and would fire on a profile that satisfies every invariant.
  Recorded here so the window is not read as a loosening: the amplitude
  bound is the triangle inequality with 50 % headroom.
- **F7b (frozen):** max over non-thirds steps of |R_step| <=
  |R_single(r = 1.4, d = max(P))| x (1 + 1e-9), where the bound is the
  chain-model single step at the cap ending on P's coarsest cell.
  Reference: **2.9545e-5 vs 2.9545e-5** (equality: the reference's largest
  step is exactly a cap step onto its coarsest cell, 171.1 um). Thirds
  pairs reported separately (reference max **1.385e-6**).
- Reported, not gated: OLD vs NEW totals as above; the OLD 9.8e-6 must not
  be quoted as "OLD reflects less" without the lambda/450 caveat.

### F8 — FDTD witness, narrow fine band (CPU, one attempt)

Fixture, all lengths from `fixtures.py` constants: coarse cell
`DC = DZ_FINE x 1.4^2 = 1.96 mm` (15.30 cells per free-space wavelength at
10 GHz; fine band 30.0). Profile A(n_b), incident from the coarse side:

```
[1.96 mm] x 140  |  1.4 mm  |  [1.0 mm] x n_b  |  1.4 mm  |  [1.96 mm] x 150
```

Both ramps sit exactly at the cap (1.96 -> 1.4 -> 1.0 and back). Built
with the builder under test: `make_band_profile(edges=[0, 275.8e-3,
275.8e-3 + n_b*1e-3, 275.8e-3 + n_b*1e-3 + 295.4e-3], cell_sizes=[1.96e-3,
1e-3, 1.96e-3], protected=[False, True, False], max_ratio=1.4)`; the test
asserts the builder's output equals the vector above to 1e-12 m per cell
(if it does not, the chain model is re-run on the builder's actual output,
that prediction is the gate, and the deviation is reported). B run
(2-run differencing reference): `[1.96 mm] x 400 + [1.0 mm] x 4` — the four
trailing fine cells pin dt to A's (verified: dt A = dt B =
**2.402765e-12 s**, both 0.99 CFL with dxy 1.5 mm) and sit beyond every
gate (their return reaches the probe at 5.055 ns). Source plane K_SRC = 85,
probe K_PRB = 100 (coarse cells, 166.6 / 196.0 mm), TE10 soft Ex source,
Gaussian-modulated sine F0 = 10 GHz, sigma_t = 64 ps, t0 = 5 sigma_t; PEC
box a = 4.5 mm, b = 30 mm, `cpml_layers = 0` (`harness.build_pec_fixture`);
**n_steps = 1200** (2.883 ns). Gates from geometry, same construction as
`w2_arm`: reflection arrival t_r = 1.047 ns (+4 sigma = 1.303 ns), last
band-internal return 1.099 ns at n_b = 4 (1.192 ns at 16), source-wall
echo t_s = 2.347 ns, far-wall t_f = 3.393 ns, gate_end = 2.091 ns
(870 steps); incident gate closes at 0.947 ns. `R_meas = |DFT_F0(A - B,
[0, gate_end])| / |DFT_F0(B, [0, t_inc_end])|` — an AMPLITUDE ratio, as in
W2.

Chain-model predictions (exact discrete solve on the vector above, dt
2.402765e-12 s, dy 1.5 mm, b 30 mm), frozen:

| n_b (cells) | band (mm) | \|R\|_model | \|R\|²_model | dB | window on \|R\|_meas |
|---|---|---|---|---|---|
| 2 | 2 | 7.4916e-3 | 5.612e-5 | -42.5 | [5.9633e-3, 9.0199e-3] |
| **4 (gate)** | 4 | **1.0141e-2** | **1.0283e-4** | -39.9 | **[8.0826e-3, 1.2199e-2]** |
| 8 | 8 | 1.1296e-2 | 1.276e-4 | -38.9 | [9.0066e-3, 1.3585e-2] |
| 16 | 16 | 1.2101e-3 | 1.464e-6 | -58.3 | [9.3810e-4, 1.4822e-3] |

Window: `|R_meas - R_model| <= 0.20 x R_model + 3e-5` (FS2_FLOOR = 3e-5 is
the W2 amplitude floor — W2's `R_meas` is an amplitude ratio — so the
window is applied in amplitude; in power that is about +/-(40 % + 6e-5 x R)).
|R|² is reported alongside. **The gate is the n_b = 4 row;** the four rows
together are the validity-domain deliverable. The law they trace, read off
the chain model before any FDTD: a single 1.96 -> 1.4 -> 1.0 ramp reflects
**5.7907e-3** (-44.7 dB, either direction); a band of width L between two
such ramps reflects about `2 x 5.79e-3 x |sin(k_g L_eff)|` — a Fabry-Perot
sum of the two opposite-sign ramp reflections. 8 cells is about lambda_g/4
(lambda_g = 34.64 mm at 10 GHz, b = 30 mm) and sits near the 2 x 5.79e-3 =
1.16e-2 maximum; 16 cells is about lambda_g/2 and lands in the null. So
"narrow" is not "worse": the narrow-band reflection is bounded by twice the
single-transition value and oscillates with band width. Rows n_b = 32
(1.5111e-3) and 64 (6.5575e-3) may be run as extra law points; reported,
not gated. All of this is one resolution (fine 30, coarse 15.3
cells/lambda) — quote the law, not the dB.

Run commands (declared) and output:

```
cd /Users/byungkwankim/Documents/rfx-nu-band && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-band \
  /Users/byungkwankim/Documents/rfx/.venv/bin/python -m validation.research.multiband_nu.w6_band_builder \
  --widths 2,4,8,16 --out validation/research/multiband_nu/results/w6_band_builder.json
```

`w6_band_builder.py` (committed BEFORE it is run) records `rfx.__file__`,
the builder's profile per width, the chain-model prediction, the gates in
ns, `R_meas`, `R_meas**2`, and the F8 verdict per row. One attempt, no
re-rolls; a fired gate is reported as fired.

## 4. Impact-sweep rule

This changes every auto-configured NU z mesh whose stack has a
protected-protected seam or an air run (cell count, dz values, dt via
dz_min, memory estimates), `make_z_profile`'s output for every input with a
descending edge, and adds two public names. For every locked test value
that moves, the NEW value is accepted only if the realized mesh now
satisfies I1-I3 (I2 under R1) and the moved value follows from that by
arithmetic; re-pin WITH provenance in the commit message. Any moved value
that cannot be justified physically => STOP and report. Values expected to
move: nz / dz_min-derived estimates on auto_configure fixtures with
adjacent dielectric layers (none of the (c) locks). Values expected NOT to
move: the (c) demo block and dz_min, every explicit-profile fixture in
`tests/_example_fidelity_lib.py`, `fixtures.py`, and the preflight NU
tests, the 1.4 z cap, the 1.3 in-plane cap, `apply_thirds_rule` and its
three tests.

## 5. Hand-off to #931 (geometry -> lattice ownership)

The thirds rule is an OWNERSHIP assumption, not a mesh-builder detail:
`_make_dz_profile` applies it at every feature boundary index — dielectric
| air AND dielectric | dielectric (core | prepreg, measured above) — and
never at a conductor, although its docstring reasons about a conductor
side. Its measured consequence on the PCB fixture: a 0.1 mm prepreg on
four 25 um cells becomes 8.333 / 16.667 / 25 / 25 / 16.667 / 8.333 um, so
**dz_min = 8.333 um** sets dt for the whole column, and under I2 the
neighbouring cores are forced to 23 cells each (nz 45 -> 105). This lane
keeps `apply_thirds_rule` and its locked tests unchanged and pays the cell
cost; whether a dielectric | dielectric seam should be split at all, and
whether dz_min should ever be a thirds sub-cell of the thinnest layer, is
#931's question. Rule from the 2026-08-23 ledger, restated so the answer
does not drift: thin copper sheets register on a NODE PLANE, never as a
17 um cell — a conductor boundary needs the node, not a split cell.

---

## Results (appended after measurement; no window above changed)

(empty at pre-declaration)
