# Accuracy and autodiff on band-builder meshes (W7) — pre-declaration

**Status:** pre-declaration. This commit precedes the instrument
(`validation/research/multiband_nu/w7_accuracy_ad.py`), the replay test and
every measurement. Every numeric window below is frozen from this commit
onward; results are appended under "Results", never edited into the
windows. One attempt per arm; a fired falsifier is a result, not a bug to
tune away.
**Class:** observable-vs-analytic on builder-made multi-band meshes, plus
autodiff integrity on the same meshes (family: W4R3 `w4r3_zdominant_cavity`,
W5 `w5_ad_consistency`, the committed cavity oracles
`test_nonuniform_xy_cavity_accuracy` / `test_nonuniform_cavity_accuracy`).
**Tree:** worktree `rfx-nu-accuracy`, branch `feat/nu-band-accuracy-ad`,
based on `feat/nu-band-profile @ 7c6e3997` (which sits on
`origin/main d990e18c`).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-accuracy/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv` and `started_utc` (the W6 convention, commit `5bf9d16b`).
Date: 2026-09-07 (KST).

## 0. Scope

What this lane sits on (established by the band-profile lane, cited not
re-derived): `rfx.make_band_profile` realizes every declared interface on a
node plane (I1, <= 1e-12 m), every adjacent ratio <= cap including seams
(I2), the exact column (I3), an optional end-cell pin (I4), on any axis
(I5); the F7 chain model and the F8 FDTD reflection witness HELD for fine
bands of 2 to 64 cells at fine 30 / coarse 15.3 cells per wavelength,
r = 1.4, PEC, 10 GHz (`results/w6_band_builder.json`, note
`20260907_nu_band_profile_predeclaration.md`).

What is NOT established and this lane measures:

1. an OBSERVABLE (a resonant frequency) computed on a builder-made
   multi-band mesh against an ANALYTIC value — (A1) a z-stratified
   dielectric cavity of the multilayer-PCB shape, (A2) in-plane dx/dy
   multi-band grading, (A3) all three axes graded at once;
2. that autodiff is intact on such meshes — (AD1) profile-vector gradients
   (mesh as design variable), (AD2) material gradients on the mesh, (AD3)
   joint dx/dy/dz gradients, (AD4) no NaN/inf across the builder's output
   family (tiny cells, narrow bands), (AD5) the min-tie convention recorded.

Scope statement, recorded so nobody reads more into the AD arms than they
say: `make_band_profile` is host-side numpy. Gradients with respect to the
layer EDGES or the target cell sizes are **not available and not claimed**.
What is claimed, and measured here, is the gradient with respect to the
REALIZED cell vector the builder returns (`dx_profile` / `dy_profile` /
`dz_profile` as `jax.grad` inputs), which is the mesh-as-design-variable
path the NU solver already carries (`test_nonuniform_gradient.py`,
`test_nonuniform_forward_grad.py`). Under that path a material is attached
to a NODE INDEX, not to a coordinate: perturbing a cell moves the node,
the node keeps its eps. The AD arms use exactly that convention.

The mesh-and-material convention this lane must record, not change: on the
NU path materials are sampled at E-node coordinates by
`Box.mask_on_coords` (volume branch half-open `[lo, hi)` per axis, host
float64) inside `rfx.runners.nonuniform.assemble_materials_nu`; there is
no subpixel averaging on that path. Section 1 measures what that does at a
node-aligned interface. Ownership of the answer is #931 (section 5).

## 1. Findings made while writing this note (zero FDTD)

### 1a. The material of a node ON an interface is decided by the last ulp of the node coordinate

Fixture: the A1 stack of section 2 (core 4.3 | thin 3.0 | core 4.3 | air,
interfaces at z = 14, 16, 30 mm, all on nodes by I1), `Simulation(...,
boundary="pec", dz_profile=<profile>)` with three `Box` entries filling
`[0,14)`, `[14,16)`, `[16,30)` mm, transverse 3 x 3 mm at dx 0.5 mm, then
`sim._build_nonuniform_grid()` and `sim._assemble_materials_nu(grid)`;
the eps column read at the central (i, j). Same three boxes, same declared
interfaces, nine profiles (UC / MB / AZ of section 2 at s = 0.5 / 1 / 2).

Realized eps at the three interface NODES (the rasterizer's own output;
"upper" = the box whose `lo` is that plane, "lower" = the box whose `hi`
is that plane; the `repr` of the node coordinate the mask compared against
the face is what decides):

| arm | s | z = 14 mm | z = 16 mm | z = 30 mm |
|---|---|---|---|---|
| UC | 0.5 | 3.0 upper (`0.01400000000000001`) | 4.3 upper (`0.01600000000000001`) | 1.0 upper (`0.030000000000000023`) |
| UC | 1 | 3.0 upper (`0.014000000000000009`) | 4.3 upper (`0.01600000000000001`) | 1.0 upper (`0.030000000000000023`) |
| UC | 2 | 3.0 upper (`0.014000000000000005`) | 4.3 upper (`0.016000000000000007`) | 1.0 upper (`0.03000000000000002`) |
| MB | 0.5 | **4.3 lower** (`0.01399999999999999`) | **3.0 lower** (`0.01599999999999999`) | **4.3 lower** (`0.029999999999999975`) |
| MB | 1 | 3.0 upper (`0.014000000000000009`) | 4.3 upper (`0.01600000000000001`) | **4.3 lower** (`0.029999999999999995`) |
| MB | 2 | 3.0 upper (`0.014`) | 4.3 upper (`0.016`) | **4.3 lower** (`0.029999999999999985`) |
| AZ | 0.5 | **4.3 lower** (`0.01399999999999999`) | **3.0 lower** (`0.015999999999999986`) | **4.3 lower** (`0.02999999999999997`) |
| AZ | 1 | 3.0 upper (`0.014`) | 4.3 upper (`0.016`) | 1.0 upper (`0.03`) |
| AZ | 2 | 3.0 upper (`0.014`) | 4.3 upper (`0.016`) | 1.0 upper (`0.03`) |

The interface is on a node in every one of the nine meshes (I1 holds to
<= 9e-17 m). The node's material is the upper box when the running sum of
the cells lands at or above the face by an ulp and the lower box when it
lands below by an ulp — the same declared geometry realizes a different
material column depending on which cells were summed to reach the plane.
This is the residue the `Box.mask_on_coords` docstring already names for
the thin branch ("still one f64 ulp of `mid` away from the tie"); for the
volume branch at a node-aligned face it is the whole answer. Recorded, not
changed (section 5).

### 1b. Consequence, from the exact discrete model: a first-order error twice the size of the grading error

For the A1 field family (Ey only, y-invariant; section 2.1) the rfx update
equations reduce EXACTLY to
`eps_k d2E_k/dt2 = c0^2 (A_x + A_z) E_k` with the same 1-D operators
`analytic_dispersion.py` uses (`inv_e` dual, `inv_h` primal, PEC ends),
`eps_k` being whatever sits at node k. The leapfrog step turns each
eigenvalue into the exact discrete frequency
(`sin(w dt/2) = (c0 dt/2) sqrt(lambda)`). Section 3.1 derives it; the
numbers below come from that model on the nine profiles, with two eps
rules at every interior node:

- **dual**: `eps_k = (eps(cell k-1) d[k-1] + eps(cell k) d[k]) /
  (d[k-1] + d[k])` — the dual-cell average, the second-order-consistent
  sampling for a tangential E component at a node-aligned planar
  interface (no cell straddles an interface, so every cell's eps is
  unambiguous);
- **production**: the column of section 1a, as assembled.

Predicted total error `f_model - f_true` of the target mode
(`f_true = 10 561 719 600.896 Hz`, section 2.1), MHz:

| arm | s | dual | production | production - dual (`e_samp`) |
|---|---|---|---|---|
| UC | 0.5 | -3.9969 | +3.9910 | +7.9878 |
| UC | 1 | -16.0053 | +0.3411 | +16.3464 |
| UC | 2 | -64.3028 | -29.4613 | +34.8415 |
| MB | 0.5 | -7.5050 | -18.3505 | -10.8455 |
| MB | 1 | -30.0973 | -77.9318 | -47.8345 |
| MB | 2 | -121.6174 | -216.6348 | -95.0174 |
| AZ | 0.5 | -7.3851 | -12.5552 | -5.1701 |
| AZ | 1 | -22.7528 | -14.9238 | +7.8290 |
| AZ | 2 | -24.3055 | -16.5254 | +7.7802 |

Fitted orders over s = 0.5 / 1 / 2 (LS slope of log|e| vs log df):
dual UC **2.004**, dual MB **2.009**; production UC **1.442**, production
MB **1.781**. The production UC ladder passes through a sign change
(+3.99, +0.34, -29.46 MHz): a first-order sampling term of one sign
cancels the second-order dispersion term of the other sign near s = 1, so
its fitted order is an accident of where the cancellation sits. With a
CONSISTENT one-sided assignment (UC: upper box at all three interfaces,
all scales) `e_samp` is +7.99 / +16.35 / +34.84 MHz — order **1.06**,
i.e. first order in the cell, +0.15 % of f at s = 1, **twice the entire
second-order grading error of the same mesh** (16.0 MHz). With the
assignment flipping between scales (MB, AZ) `e_samp` flips sign and size
with it.

The design consequence, taken here before any run: the grading gates of
section 3 are applied to arms whose eps the INSTRUMENT assembles with the
dual rule (a node-wise `MaterialArrays.eps_r` handed to the same solver
kernel, `run_nonuniform`, that the production path calls; nothing else
changed); the production
column is run as a separate, REPORTED arm at every scale with its
predicted numbers above as the reference, and the per-interface
assignment table is re-recorded from the arm's own assembled array. The
task's oracle-validity check (ii) is therefore applied to the dual UC arm;
the production UC arm is expected to fail it, and that expectation is the
#931 record, not a STOP. If the DUAL UC arm fails (ii), STOP: either the
transfer-matrix oracle, the discrete model or the solver is wrong, and
the G3 residual (section 3.2) says which of the three (model = FDTD but
order wrong: oracle; model != FDTD: solver or model).

## 2. Fixtures (every dimension an exact node at every scale; builder outputs verified)

Common to A1-A3: PEC-closed boxes (`cpml_layers = 0`, `boundary="pec"`),
lossless, mu = 1, float32 fields, harminv extraction. Scale factor s
multiplies every cell; extents never change with s. Every declared vector
below was regenerated with `make_band_profile` on this tree while writing
this note and matches the declared cells to <= 1e-12 m (the instrument
asserts this again before every arm; a mismatch is a fired I1/I3 witness,
reported, and the arm then runs on the builder's ACTUAL vector with the
model re-evaluated on it).

### 2.1 A1 — z-stratified PEC cavity, PCB-like

Stack along z (bottom to top), interfaces on nodes:

| layer | z (mm) | thickness (mm) | eps_r | MB cell (s = 1) | protected |
|---|---|---|---|---|---|
| core 1 | 0 - 14 | 14 | 4.3 | dc = 0.7 | no |
| thin | 14 - 16 | 2 | 3.0 | df = 0.5 (4 cells) | **yes** |
| core 2 | 16 - 30 | 14 | 4.3 | dc = 0.7 | no |
| air | 30 - 44 | 14 | 1.0 | dc = 0.7 | no |

`L_z = 44 mm`, `df = 0.5 s mm`, `dc = 1.4 df = 0.7 s mm` (the cap, a
single step at each thin-band seam — the W4R3 worst case), transverse
`a = 30 mm` (x), `b = 3 mm` (y), `dx = dy = df / 2 = 0.25 s mm` (the
W4R3 proportioning: transverse cells half the fine z cell). Thin/thick =
1/7: the task asked ~1/8; 1/8 exactly needs a 16 mm core, which is not a
multiple of the 0.7 s coarse cell at s = 1, and 14 mm is the nearest
thickness that is an exact multiple of BOTH dc and df at every scale
(core = 10 dc = 14 df at s = 2). Cells per wavelength at f_true and s = 1:
core (eps 4.3) 19.6 at dc, thin (eps 3.0) 32.8 at df, air 40.5 at dc;
at s = 2 half of that.

Arms per scale s in {0.5, 1, 2}:

- **UC** — uniform df everywhere: `np.full(88 / s, df)`; nz = 176 / 88 / 44.
- **MB** — `make_band_profile([0, 14, 16, 30, 44] mm, [dc, df, dc, dc],
  protected=[False, True, False, False], max_ratio=1.4)`; verified
  output `[dc] x 20/s | [df] x 4/s | [dc] x 20/s | [dc] x 20/s`, nz =
  128 / 64 / 32, max ratio 1.4000, interface error <= 9.0e-17 m, dz_min
  = df.
- **AZ** — `rfx.auto_config._make_dz_profile([(0, 14e-3, 4.3), (14e-3,
  16e-3, 3.0), (16e-3, 30e-3, 4.3)], 44e-3, dc)`: the thirds rule is
  present at every feature boundary (ratios 2.0 / 1.5 inside the splits,
  10 pairs over 1.4 by construction), the seam rule refines the cores
  (22 x 636.36 um at s = 1 instead of 20 x 700), the air run ramps
  268.5 -> 689.8 um. nz = 133 / 75 / 67, dz_min = 111.1 / 166.7 / 166.7 um
  (a thirds sub-cell of the thin layer). The auto path's thin-layer cell
  follows `min_cells_per_feature = 4` — 0.333 / 0.5 / 0.5 mm at s = 0.5
  / 1 / 2 — so the AZ ladder does not scale like the others and **no
  order is fitted to it**; each AZ scale is reported against its own
  model prediction (section 1b) and its cost (dt, cells, steps) recorded.
  Its s = 1 cells (um): `636.36 x 21, 424.24, 212.12 | 166.67, 333.33,
  500, 500, 333.33, 166.67 | 212.12, 424.24, 636.36 x 20, 424.24, 212.12 |
  268.54, 339.97, 430.39, 544.87, 689.79 x 18`.

Each of UC / MB / AZ runs twice per scale: **dual** eps (gated) and
**production** eps (reported), section 1b. Eighteen A1 runs.

Analytic reference (the oracle). Field family: **LSE^z, Ey only,
y-invariant** — the W4R3 TE_{m,0,p} family with eps(z). Ey is tangential
to every interface, so psi(z) = Ey(z) satisfies `psi'' + (eps_i k0^2 -
kx^2) psi = 0` in each layer with psi and psi' continuous at every
interface (Ey and Hx tangential, mu = 1) and psi(0) = psi(L) = 0 (PEC);
kx = m pi / a. The determinant is psi(L) of the state propagated from
(psi, psi') = (0, 1) through the four layers with the standard 2 x 2
propagator `[[cos q d, sin(q d)/q], [-q sin q d, cos q d]]`,
`q = sqrt(eps_i k0^2 - kx^2)` (complex where evanescent; the propagator
is entire in q^2 and psi(L) is real). Roots by sign scan + Brent in f.
The task's "Ez (or Hz)" source is not used: the W4R3 fixture class that
makes the graded axis carry the budget is y-invariant with b = 3 mm, which
admits only Ey; and an Ez family would put a DISCONTINUOUS component on
the interface node (D_z continuous, E_z not), so the sampling question of
section 1 would have no clean second-order rule to compare against.

Oracle self-checks the instrument must pass before any FDTD (frozen):
(i) all eps_i = 1 reproduces `f_mnp = (c0/2) sqrt((m/a)^2 + (p/L)^2)`
to **<= 1e-12 relative** for m in {1, 5}, every root in 3-16 GHz
(scratch: 1.8e-16); (i') all eps_i = 4.3 reproduces the same closed form
divided by sqrt(4.3) to <= 1e-12 (scratch: 1.9e-16).

Mode: **m = 1, p = 5** (five half-waves across the stack),
`f_true = 10 561 719 600.896 Hz` (10.5617 GHz). With a = 30 mm only the
m = 1 family exists below 12.4 GHz (the m = 5 family starts at 12.44 GHz,
m = 7 higher): neighbours 8.708377 GHz (p = 4, -17.5 %) and 12.275836 GHz
(p = 6, +16.2 %) — isolation **16.2 %** (the task's floor is 5 %; at
a = 60 mm the m = 5 / 7 families would sit 5-6 % away, which is why a is
30 mm). Source: Ey soft sources at x = a/3 and 2a/3, equal sign, same z
(the W4R3 symmetry pair: sin(m pi/3) = sin(2m pi/3) kills m = 2, 3, 4, 6;
m = 5, 7 survive but are out of band), both at **z = 16 mm** (the
thin/core-2 interface node, |psi| = 1.000, the antinode); probe Ey at
x = a/3, **z = 30 mm** (the core-2/air interface node, |psi| = 0.975).
Both planes are nodes of UC, MB AND AZ at every scale by I1 (verified
<= 2.4e-17 m); x = a/3, 2a/3 are exact nodes (40/s and 80/s cells). psi
at z = 14 mm is -0.033 (a near-null), which is why the source is not
there. j = ny // 2. Node indices k(16 mm) / k(30 mm): UC 64/120, 32/60,
16/30; MB 48/88, 24/44, 12/22; AZ 49/91, 29/53, 29/53 (s = 0.5 / 1 / 2).
Waveform: Gaussian-modulated sine at f_true, sigma_t = 200 ps, t0 = 5
sigma_t (W4R3). T = **15 ns**, harminv (`rfx.harminv.harminv`) on the
trace after 2 t0 over 7-14 GHz; the mode is the in-band (9.6-11.5 GHz)
line nearest f_true, MATCH_GUARD 3 %. Run-length invariance: the same
trace truncated to **10 ns** (a shorter run of a deterministic causal
scheme is bit-identical to a truncation) must give a frequency within
**0.1 MHz** (= E_FLOOR / 3, E_FLOOR = 0.3 MHz as W4R3) of the 15 ns value;
otherwise the arm is INCONCLUSIVE (extraction-limited), reported, and the
order fit drops it.

Cost, declared (W4R3 measured 4.2e8 cell-steps/s on this machine when
idle; assume half of that while the battery runs): s = 0.5, dt =
2.752e-13 s, 54 508 steps — UC 1.07 M cells (~280 s), MB 0.78 M (~200 s),
AZ 0.81 M cells at dt 2.284e-13 s, 65 667 steps (~250 s); twice each for
the two eps rules: ~25 min. s = 1 about 1/8 of that, s = 2 negligible.
Each (arm, scale, rule) is one resumable unit of `--arms/--scales`.

### 2.2 A2 — in-plane TM110 air cavity, two fine bands per axis

Committed oracle reused: `tests/oracle/test_nonuniform_xy_cavity_accuracy.py`
(TM110, Ez source/probe, `Simulation(boundary="pec", dx=1e-3,
dx_profile, dy_profile, dz_profile)`, `sim.run(n_steps)`,
`Result.find_resonances` over 0.6-1.5 f, nearest-to-analytic with the
separation gate `d_second > 3 d_near and > 0.5 GHz`, anti-vacuity
max/min > 2 per axis). Only the profiles change.

Profiles, `max_ratio = cap`, `boundary_cell = 1e-3` (the dx pin the
in-plane path requires), protected = [F, T, F, T, F], cell sizes
[1, 0.25, 1, 0.25, 1] mm:

- x: edges `[0, 10, 12, 26, 28, 40] mm` -> a = 40 mm;
- y: edges `[0, 8, 10, 22, 24, 34] mm` -> b = 34 mm;
- z: `[1 mm] x 10` (uniform thin, p = 0 makes z irrelevant to f).

| cap | axis | n | realized max ratio | ends | max/min | cells (mm) |
|---|---|---|---|---|---|---|
| 1.3 | x | 63 | 1.286476 | 1.0 / 1.0 | 4.0 | 1.0, 0.8809 x 8, 0.6848, 0.5323, 0.4138, 0.3216, 0.25 x 8, 0.3138, 0.3939, 0.4943, 0.6205, 0.7788, 0.9775 x 9, 0.7788, 0.6205, 0.4943, 0.3939, 0.3138, 0.25 x 8, 0.3117, 0.3887, 0.4847, 0.6044, 0.7536, 0.9397 x 9, 1.0 |
| 1.3 | y | 57 | 1.286476 | 1.0 / 1.0 | 4.0 | 1.0, 0.8481 x 6, 0.6643, 0.5203, 0.4075, 0.3192, 0.25 x 8, 0.3135, 0.3932, 0.4932, 0.6186, 0.7758, 0.973 x 7, 0.7758, 0.6186, 0.4932, 0.3932, 0.3135, 0.25 x 8, 0.3216, 0.4138, 0.5323, 0.6848, 0.8809 x 8, 1.0 |
| 1.4 | x | 60 | 1.316978 | 1.0 / 1.0 | 4.0 | 1.0, 0.9881 x 7, 0.7506, 0.5702, 0.4332, 0.3291, 0.25 x 8, 0.3288, 0.4325, 0.5689, 0.7483, 0.9843 x 10, 0.7483, 0.5689, 0.4325, 0.3288, 0.25 x 8, 0.3292, 0.4336, 0.5711, 0.7521, 0.9904 x 9, 1.0 |
| 1.4 | y | 54 | 1.316357 | 1.0 / 1.0 | 4.0 | 1.0, 0.9843 x 5, 0.7483, 0.5689, 0.4325, 0.3288, 0.25 x 8, 0.3286, 0.432, 0.5679, 0.7465, 0.9813 x 8, 0.7465, 0.5679, 0.432, 0.3286, 0.25 x 8, 0.3291, 0.4332, 0.5702, 0.7506, 0.9881 x 7, 1.0 |

(Cells quoted to 4 decimals; the instrument asserts the builder's float64
vector against its own regeneration, not against these rounded digits.)
Both caps realize the same 4:1 max/min; the 1.4-cap arm's largest step is
1.317, above the in-plane advisory threshold 1.3, so preflight's
`nu_grading_ratio_beyond_validated_cap` and the constructor's "adjacent
cell ratio" warning are EXPECTED on that arm and are reported, not gated.
Uniform control: `dx = dy = dz = 1 mm`, 40 x 34 x 10 cells — exact
extents (a and b are integers in mm), unlike the committed test's
1.0083/1.0096 mm control. Analytic: `TM110 = 5.786173 GHz`, nearest
Ez-coupled neighbours TM210 8.695 GHz and TM120 9.581 GHz (50 %
isolation). Source (a/3, b/3, d/2) ez `GaussianPulse(f0 = f110,
bandwidth 0.8)`, probe (2a/3, 2b/3, d/2) ez — snapped by the Simulation to
the nearest node (recorded); for m = n = 1 there is no interior null.
n_steps **8000** (the committed value) and **12000** (invariance) on
every A2 arm; graded dt 5.7485e-13 s (8000 steps = 4.6 ns), uniform dt
1.9066e-12 s.

### 2.3 A3 — TM111, all three axes multi-band from the builder

Committed oracle reused: `tests/oracle/test_nonuniform_cavity_accuracy.py`
(TM111, ez source/probe at (a/3, b/3, d/4) / (2a/3, 2b/3, d/4), the same
extraction and separation gate). Profiles, cap 1.4, `boundary_cell =
1e-3`, cell sizes [1, 0.25, 1] mm, protected [F, T, F]:

| axis | edges (mm) | n | realized max ratio | cells (mm) |
|---|---|---|---|---|
| x | 0, 19, 21, 40 | 50 | 1.318010 | 1.0, 0.9943 x 16, 0.7544, 0.5724, 0.4343, 0.3295, 0.25 x 8, 0.3295, 0.4343, 0.5724, 0.7544, 0.9943 x 16, 1.0 |
| y | 0, 16.5, 18.5, 35 | 46 | 1.308969 | 1.0, 0.9607 x 14, 0.7339, 0.5607, 0.4284, 0.3272, 0.25 x 8, 0.3272, 0.4284, 0.5607, 0.7339, 0.9607 x 14, 1.0 |
| z | 0, 17, 19, 36 | 46 | 1.317812 | 1.0, 0.9936 x 14, 0.754, 0.5721, 0.4342, 0.3295, 0.25 x 8, 0.3295, 0.4342, 0.5721, 0.754, 0.9936 x 14, 1.0 |

a x b x d = 40 x 35 x 36 mm (the committed z-oracle's transverse box);
grid 51 x 47 x 47 nodes, dt 4.7664e-13 s. Uniform control 1 mm on all
axes (40 x 35 x 36 cells, dt 1.9066e-12 s). Analytic `TM111 = 7.051389
GHz`; nearest neighbour TM110 5.691 GHz (19.3 %), then TM211 9.584 GHz.
n_steps **8000 and 12000** on every A3 arm (3.8 / 5.7 ns graded; the
committed z-test's 8000 steps were 6.2 ns at its dt — the invariance check
is what protects the extraction here, and T is recorded).

### 2.4 AD1 — profile gradient on the A1 MB dz vector (W5 pattern)

Mesh: the A1 MB vector at s = 2, `[1.4 mm] x 10 | [1.0 mm] x 2 |
[1.4 mm] x 10 | [1.4 mm] x 10` (32 cells), on a small transverse box
`domain_xy = (6 mm, 3 mm)`, `dx = 0.5 mm` (= df/2 at s = 2, as A1),
`cpml_layers = 0`; dt 1.1008e-12 s. eps per node = the PRODUCTION column
of section 1a for MB s = 2 (3.0 at k = 10, 11; 4.3 at k = 0-9 and 12-22;
1.0 at k = 23-32), held fixed BY INDEX while dz varies. Ey source at
(i = 6, j = 3, k = 12) (x = 3 mm, z = 16 mm), Gaussian `exp(-((t -
15 dt)/(5 dt))^2)` (the W5 waveform); Ey probe at (6, 3, 10) (z = 14 mm,
across the thin band); `N_STEPS = 120` (132 ps: the pulse crosses the
band, ~11 ps, and reflects within the cores). Loss `L = sum(ts^2)`.
`jax.grad(L)(dz)` on the f32 path vs central FD with `h = 1e-3 dz[k]`
(W5's FD_REL_H) on every cell. x64-context arm (`tests._x64_compat
.enable_x64`) run and reported as in W5, not gated.

### 2.5 AD2 — material gradient d L / d eps_thin on the A1 MB mesh

Same grid, source, probe, N_STEPS and loss as AD1. `eps_thin` is a scalar
set on the thin-layer node set = the nodes carrying eps 3.0 in the
assembled production column (k = 10, 11 at s = 2) via
`eps.at[..., thin_nodes].set(eps_thin)`; `jax.grad` w.r.t. that scalar at
eps_thin = 3.0 vs central FD with `h = 0.015` (0.5 % — the committed
material-gradient convention `test_forward_nonuniform_grad_eps_matches_fd`
uses h = 1e-2 on alpha = 2.0).

### 2.6 AD3 — joint (dx, dy, dz) gradient on the A3 mesh

The three A3 builder vectors (2.3) as a tuple of `jnp.float32` inputs to
`make_nonuniform_grid(domain_xy, dz_profile=dz, dx=1e-3, dx_profile=dx,
dy_profile=dy, cpml_layers=0)` (the tracer path; `dx_profile[0] ==
dx_profile[-1] == dx` holds by I4), vacuum materials, `run_nonuniform`.
Ez source at the node nearest (20, 17.5, 18) mm = (i, j, k) = (25, 23,
23) — the centre of all three fine bands; Ez probe at (28, 23, 23)
(0.75 mm away along x); W5 waveform; `N_STEPS = 120` (57 ps, reaching the
ramps and the coarse cells on every axis). `jax.grad` w.r.t. the tuple vs
central FD (`h = 1e-3 d[k]`) on every cell of every axis (142 cells,
284 forward runs of ~1.1e5 cells x 120 steps).

### 2.7 AD4 — robustness smoke over 20 seeded builder stacks

`rng = np.random.default_rng(20260907)`, 20 stacks drawn in this order
per stack: `n_layers = rng.integers(1, 7)`; thicknesses log-uniform
20 um - 3 mm (`10 ** rng.uniform(log10(20e-6), log10(3e-3), n_layers)`);
target fraction log-uniform 0.05-0.33 of the layer (so every layer has
>= 3 cells); `protected = rng.random(n) < 0.5`; `cap = rng.choice([1.2,
1.3, 1.4])`; `eps = rng.uniform(1, 6, n)`; `pin = rng.random() < 0.5`,
and when pinned both end segments are forced free and `boundary_cell =
min(target[0], target[-1], min(t[0], t[-1]) * min(1/4, (cap - 1)/(1.1
cap)))` (the builder fuzz's feasibility rule). Verified on this tree: all
20 build; nz from 6 (stack 17) to 580 (stack 7), dz_min down to
**1.847 um** (stack 6: a 26 um layer next to a 31 um one at cap 1.4);
layer counts 5, 4, 5, 6, 1, 5, 4, 5, 6, 3, 3, 1, 5, 6, 3, 3, 2, 1, 5, 3.
Each stack: `dz_profile` = the builder's vector, `domain_xy = (3 mm,
3 mm)`, `dx = 1 mm`, `cpml_layers = 0`; eps per node from the half-open
rule on the unperturbed profile, fixed by index; Ez source at (1, 1,
k_src = nz // 2), probe at (1, 1, k_src - 1 if >= 1 else k_src + 1);
`N_STEPS = 20`; forward trace and `jax.grad(sum(ts^2))(dz)` on the f32
path.

### 2.8 AD5 — the min-tie convention

`make_nonuniform_grid` sets dt from `jnp.min` of each axis's padded
profile. On a builder profile the minimum is attained by EVERY cell of a
fine band (2 cells in AD1, 8 per axis in AD3): the loss is not smooth in
those cells (a positive perturbation of one tied cell leaves dt unchanged,
a negative one moves it), central FD returns the average of two different
one-sided slopes, and the AD value is a convention (JAX splits the
cotangent of a tied `min` equally over the tied entries; the committed
test docstring describes an argmin pick — the instrument prints the
per-tied-cell `g_ad` so the realized convention is on record). Rule for
AD1 and AD3, declared here: **tied-minimum cells are excluded from the
dominant-cell comparison**, and for each of them `g_ad`, the one-sided
`FD+` and `FD-` are reported side by side. W5 avoided the question with
1 % jitter; that is not available here because the vector under test IS
the builder's.

## 3. Falsifiers (frozen; derivations stated; tolerances never widened after measurement)

### 3.1 The derivation instrument (first principles, no FDTD)

For A1 (Ey, y-invariant) the discrete problem is, on the interior z nodes
1..N-1 of the padded profile (`#562` trailing bounding node, `inv_h[-1] =
0`), the generalized symmetric eigenproblem
`(-L + mu_x DD) Z = lambda (DD Eps) Z` with `L` the `inv_h`-weighted
second difference, `DD = diag(dual spacing (d[k-1] + d[k])/2)`, `Eps =
diag(eps_k)`, `mu_x` the first eigenvalue of the uniform-x operator
(`analytic_dispersion.operator_eigenvalues`), and `f = asin(c0 dt
sqrt(lambda)/2) / (pi dt)`. With eps = 1 it is `analytic_dispersion.decompose`
exactly. Decomposition: `e_z = f(lambda_discrete) - f(lambda_cont(mu_x))`
where `lambda_cont(mu_x)` is the transfer-matrix root with `kx^2 := mu_x`;
`e_x = f(lambda_cont(mu_x)) - f(lambda_true)`; `e_t = f(lambda_true) -
f_true`; `e_samp = f(production eps) - f(dual eps)`. For A2/A3 (Ez only)
`lambda = mu_x + mu_y (+ mu_z)` from the per-axis operators on the builder
vectors (vacuum), the CORE-C2 metric structure being the same on every
axis. Why this model is trusted for windows: on W4R3 it predicted the
MB/UC ratio to **+0.03 / -0.14 / +0.32 / +1.30 %** at s = 2 / 1 / 0.5 /
0.25 (residual <= 33 kHz), and on the COMMITTED xy-oracle fixture it
predicts **-0.02775 %** against the committed measured 0.0282 %. The
instrument must reproduce every model number quoted in this note to
1e-7 relative in f before any FDTD (`--selfcheck`).

### 3.2 A1

- **A1-O (oracle validity, i / i'):** closed forms reproduced to
  <= 1e-12 relative (section 2.1). Fail => STOP, the oracle is wrong.
- **G3 (model residual, every arm, every scale, both eps rules):**
  `|f_meas - f_model| <= 0.15 MHz` (W4R3's window; measured there <= 33
  kHz). Fail on a dual arm => the solver does not do what its own
  operators say (STOP, solver); fail on a production arm only => the
  assembled column differs from the one the model was given (the arm
  re-reads its column and re-evaluates; still failing => STOP).
- **G1 (z-fraction):** `|e_z| / (|e_z| + |e_x| + |e_t|) >= 0.80` on UC and
  MB dual at every scale (model 0.964 / 0.980). **G2 (grading share):**
  `|e_z(MB) - e_z(UC)| / |e_tot(MB)| >= 0.20` (model 0.467-0.471). Either
  failing => FIXTURE-INVALID, no claim.
- **A1-O (ii), the task's oracle-validity check, on the DUAL UC arm:**
  fitted order `p_uc` in **[1.8, 2.2]** over s = 0.5 / 1 / 2 (fit floor
  0.3 MHz, >= 3 surviving points; model 2.004). Fail => STOP and report
  which of oracle / model / solver, by G3 as in section 1b.
- **A1-F1 (MB order):** `p_mb >= 1.8` on the dual MB arm (model 2.009);
  anomaly flag if `p_mb > 2.4`. Fires => the builder's cap-1.4 seam at a
  4-cell dielectric band costs an order of convergence: the band-builder
  claim is NOT supported for dielectric bands; record, STOP the A1 arm.
- **A1-F2 (error amplitude at matched finest cell):** `rho = 10 ^
  (b_mb(h1) - b_uc(h1))`, the ratio of the two fitted lines evaluated at
  h1 = df(s = 1), **<= 2.0**. Derivation: on a Yee axis the leading
  dispersion error of a mode is proportional to the length-weighted sum
  of `k_z^4 d^2`, so at matched finest cell a graded mesh's error is at
  most `r_max^2 = 1.96` times the uniform one on the graded part; the
  shared e_x and e_t, the four fine cells and the mode's weighting bring
  the exact model to **1.883** (1.878 / 1.881 / 1.891 at the three
  scales). 2.0 is the task's number, taken from the F-S4 precedent (W4R3
  measured 1.56 at the same cap with 56 % of the axis fine; 2.0 leaves
  that 28 % headroom) and sits 6.2 % above the model, where the model's
  own worst deviation from measurement on W4R3 was 1.3 % and the harminv
  floor contributes ~1.5 % at the fitted amplitude. A W4R3-class metric
  defect on one transition node moved that fixture's MB error by 120 %
  at s = 1; here it would take rho past 3. Fires => grading-specific
  error beyond the coarse cells' own dispersion; record, STOP.
- **A1-R (reported, not gated):** the production arms — per scale
  `f_meas`, `e_samp,meas = f_meas(production) - f_meas(dual)` against the
  section 1b table, the fitted order of the production ladders (expected
  1.44 UC / 1.78 MB, both meaningless as orders, recorded as such), the
  re-recorded interface-assignment table; the AZ arms — `f_meas` vs model
  at each scale, nz, dz_min, dt, steps, cells, wallclock; cost of MB vs UC
  (cells, steps, wallclock) at every scale.

### 3.3 A2

Derivation: the exact model predicts graded **-0.02146 %** (cap 1.3) and
**-0.02766 %** (cap 1.4) against **-0.01142 %** uniform — differences
**-0.0100 / -0.0162 pt**; the committed single-band envelope was 0.0282 %
(gate 0.05 %) with 0.0198 pt over its uniform, and its n_steps scatter
0.0071 pt. Two bands per axis double the transitions; the model already
contains them. Windows, each arm (cap 1.3 and cap 1.4), each n_steps:

- **A2-F1:** `|f_meas - f_110| / f_110 <= 0.10 %` (3.6x the model at
  cap 1.4; 2x the committed gate for twice the transitions).
- **A2-F2:** `|err_graded - err_uniform| <= 0.05 pt` (3.1x the model at
  cap 1.4; the task's number, derived as 2.5x the committed 0.0198 pt
  single-band excess).
- **A2-V:** separation gate and anti-vacuity as committed;
  `|f(8000) - f(12000)| / f_110 <= 0.017 %` (1/3 of A2-F2) else
  INCONCLUSIVE (extraction).
- Reported: uniform arm error (model -0.0114 %; committed 0.0084 % at
  5.04 GHz), the cap-1.4 advisory text, the snapped source/probe indices,
  the model residual per arm (no gate — the Simulation path adds source
  normalization the model does not carry; the residual is recorded as
  the model's validity class on this path).

### 3.4 A3

Derivation: model graded **-0.03488 %** (e_x -0.0088, e_y -0.0144, e_z
-0.0136, e_t +0.0019 pt) vs uniform **-0.00098 %**, difference
**-0.0339 pt**. The task's rule — the loosest single-axis window times
two — gives:

- **A3-F1:** `|err_graded| <= 0.20 %` (2 x A2-F1; 5.7x the model).
- **A3-F2:** `|err_graded - err_uniform| <= 0.10 pt` (2 x A2-F2; 2.9x the
  model).
- **A3-V:** separation, anti-vacuity (max/min = 4 on every axis),
  `|f(8000) - f(12000)| / f_111 <= 0.033 %` else INCONCLUSIVE.

### 3.5 AD1 / AD2 / AD3 / AD4

- **AD1-F:** on dominant cells (`|g_fd| > 5 %` of the largest `|g_fd|`
  over NON-tied cells), sign agreement and `|g_ad - g_fd| / |g_fd| <=
  0.15` (the committed NU AD convention,
  `test_grad_wrt_dz_profile_matches_fd` and W5; W5 measured 1.1e-4).
  Tied-minimum cells excluded and reported (AD5).
- **AD2-F:** `|g_ad - g_fd| / |g_fd| <= 0.05` — the committed
  material-gradient convention is tighter than 15 % and is adopted:
  `test_forward_nonuniform_grad_eps_matches_fd` gates at 5 % ("the house
  AD-vs-FD standard, cf. MSL / waveguide-flux gradient tests").
- **AD3-F:** AD1-F per axis (dominant cells per axis, tied-minimum cells
  of that axis excluded and reported), 15 %.
- **AD4-F:** over all 20 stacks, `dt > 0`, the forward trace finite, the
  gradient finite — **zero** non-finite values; `max |g| > 0` on each
  stack reported (a constant-zero gradient is finite and would hide a
  severed trace; reported so it cannot hide, not gated because a
  20-step trace on a 6-cell stack may legitimately barely move).
- **AD5:** the tie table (section 2.8) is a knowledge output; no gate.
- Any `TracerArrayConversionError` or exception on the tracer path with
  `cpml_layers = 0` is a fired AD witness (the committed sentinels use
  cpml_layers 2-4): recorded as such, the arm STOPs, no re-run with a
  different cpml setting.

## 4. Declared commands, outputs, tests

Instrument `validation/research/multiband_nu/w7_accuracy_ad.py`
(committed BEFORE it is run; `--arms` from {a1, a2, a3, ad1, ad2, ad3,
ad4}, `--scales` for a1, `--out` merged per arm so a ladder can be split
across calls and resumed; `--selfcheck` runs the oracle checks, the
builder-vector assertions and the model table without FDTD and refuses to
run any arm if one fails). Output
`validation/research/multiband_nu/results/w7_accuracy_ad.json`, one file,
per-arm keys, each run carrying `rfx_file`, `git_sha`, `git_dirty`,
`argv`, `started_utc`, the realized profile, the assembled interface eps,
dt, n_steps, cells, wallclock, the harminv line list, `f_meas`, every
model term, every gate value and verdict.

```
cd /Users/byungkwankim/Documents/rfx-nu-accuracy && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-accuracy \
  /Users/byungkwankim/Documents/rfx/.venv/bin/python -c "import rfx; print(rfx.__file__)"
PYTHONPATH=... python -m validation.research.multiband_nu.w7_accuracy_ad --selfcheck --out validation/research/multiband_nu/results/w7_accuracy_ad.json
PYTHONPATH=... python -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --scales 2,1 --out validation/research/multiband_nu/results/w7_accuracy_ad.json
PYTHONPATH=... python -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --scales 0.5 --out validation/research/multiband_nu/results/w7_accuracy_ad.json
PYTHONPATH=... python -m validation.research.multiband_nu.w7_accuracy_ad --arms a2,a3 --out validation/research/multiband_nu/results/w7_accuracy_ad.json
PYTHONPATH=... python -m validation.research.multiband_nu.w7_accuracy_ad --arms ad1,ad2,ad3,ad4 --out validation/research/multiband_nu/results/w7_accuracy_ad.json
```

(`PYTHONPATH=...` abbreviates the pinned path above; every call from the
worktree root; Bash timeouts up to 600 s per call, the s = 0.5 A1 units
split further with `--rules dual|production` if a single call would not
fit.) Order: selfcheck -> A1 s = 2, 1 -> A1 s = 0.5 -> A2 -> A3 -> AD1-4
-> results commit(s). One attempt per (arm, scale, rule); if an instrument
defect forces a re-run, both runs are kept in the JSON and the note says
so.

Replay test `tests/unit/nonuniform/test_band_accuracy_ad_replay.py`
(no FDTD, fast lane): regenerates every declared vector with the builder
on the tree and compares cell for cell (1e-12 m) with the JSON; re-runs
the oracle checks (i)/(i'); recomputes every model frequency from the
JSON's profiles and columns to 1e-9 relative; re-evaluates every frozen
rule of section 3 from the JSON's measured numbers (orders, rho, G1-G3,
A2/A3 windows, AD windows, the tie table's presence); pins the section 1a
interface-assignment table and the `e_samp` first-order law on the UC
production ladder (order 1.06 to 0.1). Plus one fast live AD test in the
same file: AD1 on the s = 2 vector with `N_STEPS = 40`, finite gradient
and the two largest non-tied cells within 15 % of central FD (about 10 s).
Tolerances written in the test before its first run.

## 5. Impact-sweep rule and the #931 note

This lane adds an instrument, a results JSON, a test and this note. It
changes NO `rfx/` source. If bring-up finds an `rfx/` defect (the candidate
already visible: the tracer path of `make_nonuniform_grid` with
`cpml_layers = 0`, never exercised by the committed sentinels), the fix is
a separate commit BEFORE any measurement, with the band-profile note's F6
rule applied verbatim: every locked value that moves is listed old -> new
and accepted only with a physical justification and provenance in the
commit message; a moved value that cannot be justified => STOP and report.
Batteries for that case: `tests/unit/nonuniform tests/unit/autodiff/
test_nonuniform_gradient.py tests/unit/autodiff/test_nonuniform_forward_grad.py
tests/unit/grid/test_auto_config.py -q -o addopts="" -m "not gpu and not
slow"` and `tests/contracts -q -o addopts="" -m "not gpu"`.

**#931 (geometry -> lattice ownership), two records from this note, both
recorded and neither changed here:**

1. The band-profile note handed #931 the thirds rule's cost (a
   dielectric | dielectric seam split for a conductor rationale; dz_min a
   thirds sub-cell of the thinnest layer). The AZ arm here carries that
   cost on record: dz_min 166.7 um against the MB arm's 500 um at s = 1,
   dt 4.005e-13 vs 5.504e-13 s, nz 75 vs 64, for the same stack.
2. New: at a node-aligned interface the E node ON the plane takes the
   material of the upper OR the lower box depending on the last ulp of
   the running cell sum (section 1a — nine meshes, three assignments
   patterns), and for a tangential E component either one-sided choice is
   a first-order error equal to moving the interface by half the adjacent
   dual cell (section 1b: +16.3 MHz = +0.15 % at s = 1 on the uniform
   mesh, twice the second-order grading error). The second-order rule is
   the dual-cell average, which no production path applies. The band
   builder puts every interface on a node (I1) exactly as the rasterizer's
   own docstring asks of registered sheets; what the node's material then
   is, is the lattice's question, not the builder's — #931's. This lane
   measures both rules and quotes the law; it does not touch
   `Box.mask_on_coords`, `rasterize_geometry` or `apply_thirds_rule`.

## 6. Where this note departs from the task as written, and why

- Thin/thick = 1/7, not 1/8 (exact multiples of dc AND df at every scale).
- Field family LSE / Ey, not Ez or Hz (the y-invariant W4R3 fixture class;
  a continuous component on the interface node).
- a = 30 mm (the m = 5, 7 families out of band: 16 % isolation, half the
  transverse cells); b = 3 mm, dx = dy = df/2 as W4R3.
- The grading gates run on instrument-assembled dual-cell eps; the
  production sampling is a reported arm with a pre-declared expectation
  (section 1). The task's check (ii) is applied to the dual UC arm.
- The AZ arm is reported per scale, no order fitted (its thin-layer cell
  does not scale with s).
- Run-length invariance by truncating the 15 ns trace to 10 ns (bit-
  identical to a shorter run), not a second FDTD run.
- The 2.0 ratio window is kept and sits 6.2 % above the exact model
  (1.883); the derivation is stated rather than the number moved.
- AD1/AD2 run the A1 MB s = 2 dz vector on a 6 x 3 mm transverse box (the
  W5 pattern), not the full A1 box; AD2 uses the committed 5 % material
  convention, AD1/AD3 the committed 15 % profile convention.
- Tied-minimum cells are excluded from the AD1/AD3 dominant-cell
  comparison and reported one-sided (no jitter: the vector under test is
  the builder's).

## Results

Measured 2026-09-07 06:42-06:53 UTC on the shared machine (load ~10, a
test battery running in another worktree). Instrument commit `5fdfced9`
(A1), `4c5f6476` (A2/A3), `15a59c33` (AD1-4); `rfx.__file__ =
/Users/byungkwankim/Documents/rfx-nu-accuracy/rfx/__init__.py` printed by
every call; no `rfx/` source touched. The `git_dirty = true` on the A1
rows is the results JSON itself being written into the tree between
commits; nothing else was modified. One attempt per unit, no unit
re-run, no `--force` used, no window edited. Everything below is read
from `validation/research/multiband_nu/results/w7_accuracy_ad.json`.

Selfcheck (auto-run before every arm): oracle (i) worst 1.81e-16, (i')
1.88e-16 relative; every model row of this note reproduced; all_pass.

### A1 — z-stratified cavity, 15 ns harminv, both eps rules

Per unit: measured error `f_meas - f_true`, the exact discrete model,
their residual (gate G3, window 0.15 MHz), run-length invariance
`|f(15 ns) - f(10 ns)|` (window 0.1 MHz), z-fraction. All in MHz.

| unit | f_meas GHz | err | model | resid | inv | z-frac | nz | cells | steps | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| uc 2 dual | 10.497397 | -64.322 | -64.303 | -0.019 | 0.036 | 0.964 | 44 | 19215 | 13627 | 0.9 |
| uc 2 prod | 10.532255 | -29.465 | -29.461 | -0.004 | 0.001 | 0.928 | 44 | 19215 | 13627 | 0.8 |
| uc 1 dual | 10.545680 | -16.040 | -16.006 | -0.035 | 0.066 | 0.964 | 88 | 139997 | 27254 | 6.7 |
| uc 1 prod | 10.562090 | +0.370 | +0.341 | +0.029 | 0.059 | 0.257 | 88 | 139997 | 27254 | 6.0 |
| uc 0.5 dual | 10.557741 | -3.979 | -3.997 | +0.018 | 0.035 | 0.964 | 176 | 1066425 | 54508 | 87.8 |
| uc 0.5 prod | 10.565731 | +4.011 | +3.991 | +0.020 | 0.057 | 0.961 | 176 | 1066425 | 54508 | 88.1 |
| mb 2 dual | 10.440043 | -121.677 | -121.618 | -0.059 | 0.009 | 0.980 | 32 | 14091 | 13627 | 0.6 |
| mb 2 prod | 10.345074 | -216.646 | -216.635 | -0.011 | 0.059 | 0.989 | 32 | 14091 | 13627 | 0.6 |
| mb 1 dual | 10.531626 | -30.093 | -30.098 | +0.004 | 0.023 | 0.980 | 64 | 102245 | 27254 | 4.6 |
| mb 1 prod | 10.483764 | -77.956 | -77.932 | -0.024 | 0.014 | 0.992 | 64 | 102245 | 27254 | 4.6 |
| mb 0.5 dual | 10.554219 | -7.501 | -7.505 | +0.005 | 0.005 | 0.980 | 128 | 777225 | 54508 | 58.4 |
| mb 0.5 prod | 10.543340 | -18.380 | -18.351 | -0.029 | 0.058 | 0.992 | 128 | 777225 | 54508 | 56.4 |
| az 2 dual | 10.537389 | -24.331 | -24.306 | -0.025 | 0.061 | 0.976 | 67 | 29036 | 30130 | 2.2 |
| az 2 prod | 10.545163 | -16.556 | -16.525 | -0.031 | 0.075 | 0.965 | 67 | 29036 | 30130 | 1.9 |
| az 1 dual | 10.538970 | -22.749 | -22.753 | +0.004 | 0.039 | 0.985 | 75 | 119548 | 37457 | 7.2 |
| az 1 prod | 10.546804 | -14.915 | -14.924 | +0.008 | 0.023 | 0.978 | 75 | 119548 | 37457 | 6.5 |
| az 0.5 dual | 10.554337 | -7.383 | -7.385 | +0.002 | 0.037 | 0.986 | 133 | 807350 | 65667 | 72.8 |
| az 0.5 prod | 10.549165 | -12.554 | -12.555 | +0.001 | 0.004 | 0.991 | 133 | 807350 | 65667 | 75.6 |

Gates and falsifiers (measured vs window):

- **A1-O (i)/(i') HELD**: 1.81e-16 / 1.88e-16 <= 1e-12.
- **G3 HELD** on all 18 units: worst residual 0.059 MHz (mb 2 dual) on the
  dual rule, 0.031 MHz (az 2 prod) on production; window 0.15 MHz.
- **Run-length invariance HELD** on all 18: worst 0.075 MHz (az 2 prod);
  window 0.1 MHz. Every unit `valid` at 15 ns and at 10 ns.
- **G1 HELD**: min z-fraction over UC/MB dual 0.964 (>= 0.80; model 0.964).
  The 0.257 on `uc 1 prod` is the production arm where the first-order
  sampling term nearly cancels the grading error (+0.37 MHz total); it is
  a reported arm and not in the G1 set.
- **G2 HELD**: min grading share 0.467 (>= 0.20; model 0.467-0.471).
- **A1-O (ii) HELD**: dual UC fitted order **p_uc = 2.007**, window
  [1.8, 2.2] (model 2.004); 3 points, all above the 0.3 MHz floor
  (smallest 3.98 MHz).
- **A1-F1 HELD**: dual MB fitted order **p_mb = 2.010** >= 1.8 (model
  2.009); no anomaly (< 2.4).
- **A1-F2 HELD**: **rho = 1.884** at h = df(s = 1) <= 2.0 (exact model
  1.883); per-scale MB/UC ratios 1.885 / 1.876 / 1.892 at s = 0.5 / 1 / 2
  (model 1.878 / 1.881 / 1.891).
- **A1-R (reported)**: measured `e_samp = f(prod) - f(dual)` in MHz vs the
  section 1b table: UC +7.990 / +16.410 / +34.857 (model +7.988 / +16.347
  / +34.842) at s = 0.5 / 1 / 2; MB -10.879 / -47.862 / -94.969 (model
  -10.845 / -47.834 / -95.017); AZ -5.171 / +7.834 / +7.774 (model -5.170
  / +7.829 / +7.780). Production ladder fitted orders 1.438 UC and 1.780
  MB (expected 1.44 / 1.78; not orders of anything, recorded as such).
  The interface-assignment tables re-recorded by the instrument matched
  section 1a on all nine meshes. AZ: dz_min 0.167 / 0.167 / 0.111 mm at
  s = 2 / 1 / 0.5 (nz 67 / 75 / 133), error -24.3 / -22.7 / -7.4 MHz
  dual — the thin-layer cell does not scale with s, so the AZ error
  does not follow the s^2 law; residual vs model <= 0.031 MHz. Cost MB
  vs UC at matched s: cells 0.73x (14091 / 19215, 102245 / 139997,
  777225 / 1066425), steps identical (same dt, dz_min = df on both),
  wallclock 0.67-0.69x.

Reading: on every mesh the solver did what its own operators say to
<= 0.06 MHz (6e-6 relative); the grading error of the cap-1.4 four-cell
dielectric band is second order with the same order as the uniform mesh,
and its amplitude at matched finest cell is 1.88x the uniform one, which
is the Yee `sum(k_z^4 d^2)` bound with r^2 = 1.96 minus the shared terms
— the law of section 3.2, measured. The first-order production-sampling
term (section 1) is confirmed at all three scales to <= 0.06 MHz against
its exact model; #931 stands.

### A2 — in-plane two-band TM110, 5.786173 GHz

| arm | steps | f GHz | err % | model % | err_uniform % | diff pt |
|---|---|---|---|---|---|---|
| cap 1.3 | 8000 | 5.785435 | -0.01276 | -0.02146 | -0.01083 | -0.0019 |
| cap 1.3 | 12000 | 5.785167 | -0.01738 | -0.02146 | -0.01100 | -0.0064 |
| cap 1.4 | 8000 | 5.785077 | -0.01893 | -0.02766 | -0.01083 | -0.0081 |
| cap 1.4 | 12000 | 5.784807 | -0.02360 | -0.02766 | -0.01100 | -0.0126 |
| uniform | 8000 | 5.785546 | -0.01083 | -0.01142 | | |
| uniform | 12000 | 5.785537 | -0.01100 | -0.01142 | | |

- **A2-F1 HELD** at both caps and both step counts: worst |err| 0.0236 %
  (cap 1.4, 12000) <= 0.10 %.
- **A2-F2 HELD**: worst |diff| 0.0126 pt (cap 1.4, 12000) <= 0.05 pt.
- **A2-V HELD**: invariance 0.0046 % (cap 1.3) and 0.0047 % (cap 1.4)
  <= 0.017 %; separation and anti-vacuity pass on every unit. Snapped
  source (13, 11, 5) and probe (27, 23, 5) as declared. Wallclock 0.8-1.9
  s per unit. The cap-1.4 in-plane advisory fired on the 1.4 arm as the
  preflight says it will (max ratio 1.318 x / 1.309 y > 1.3); recorded,
  accepted, this note is the report it asks for.
- Model residual (reported, no gate): +0.0087 / +0.0041 pt at cap 1.3,
  +0.0087 / +0.0041 pt at cap 1.4, +0.0006 / +0.0004 pt uniform — the
  Simulation path sits 0.004-0.009 pt above the bare-operator model at
  8000 / 12000 steps; the step-count dependence (0.0046 %) is of the same
  size as the committed single-band scatter (0.0071 pt).

### A3 — TM111, all three axes multi-band, 7.051389 GHz

| arm | steps | f GHz | err % | model % | diff pt |
|---|---|---|---|---|---|
| graded | 8000 | 7.048539 | -0.04042 | -0.03488 | -0.0393 |
| graded | 12000 | 7.048516 | -0.04074 | -0.03488 | -0.0390 |
| uniform | 8000 | 7.051311 | -0.00111 | -0.00098 | |
| uniform | 12000 | 7.051266 | -0.00175 | -0.00098 | |

- **A3-F1 HELD**: |err| 0.0407 % <= 0.20 % (model 0.0349 %).
- **A3-F2 HELD**: |diff| 0.0393 pt <= 0.10 pt (model 0.0339 pt).
- **A3-V HELD**: invariance 0.0003 % <= 0.033 %; separation and
  anti-vacuity (max/min = 4 on every axis) pass. Model residual -0.0055 /
  -0.0059 pt graded. Wallclock 1.1-2.3 s per unit.

### AD1-AD5 — autodiff on the builder's meshes

- **AD1 HELD** (f32, 120 steps, 32-cell A1 MB s = 2 vector, production
  eps by index, pattern verified): 6 dominant non-tied cells (8, 9, 12,
  13, 14, 15; |g_fd| 16-270), worst rel err **1.31e-3**, median 8.5e-5,
  signs agree; window 0.15. x64 context (reported): worst 1.06e-3, same
  dominant set. AD5 tie table for the two 1.0 mm cells: k = 10 g_ad
  +120.0 between FD+ +144.3 and FD- +95.4; k = 11 g_ad +26.3 between
  FD+ +49.3 and FD- +2.65 — the JAX tied-min convention lands between
  the two one-sided slopes on both cells. Wallclock 2.0 s.
- **AD2 HELD**: g_ad 0.012451 vs g_fd 0.012499, rel err **3.8e-3** <= 0.05,
  signs agree; h = 0.015 on eps_thin = 3.0. Wallclock 1.2 s.
- **AD3 FIRED** (f32, 120 steps, A3 vectors, 142 cells, 284 forward runs,
  wallclock 7.0 s). Per axis, dominant non-tied cells / tied cells /
  worst rel err / sign agreement: x 3 / 8 / **0.0043** / yes (HELD);
  y 18 / 8 / **0.522** / yes (FIRED); z 12 / 8 / **0.992** / yes (FIRED).
  All 142 AD values finite; every sign agrees. The arm is STOPPED as
  declared; not re-run, not re-tuned, no x64 arm was declared for AD3
  and none was run.

  What the recorded data say about the fired rows (arithmetic on the
  JSON, no new FDTD): `loss0 = 0.13063` and one f32 ulp of it is
  1.49e-8, so a central FD with `h = 1e-3 d_k` cannot resolve a slope
  finer than `ulp / (2h)` = 7.45e-3 on a 0.994 mm cell (3.0e-2 on a
  0.25 mm cell). The non-tied `g_fd` values on the coarse cells are in
  fact exact multiples of that quantum (7.49e-3, 1.4986e-2,
  2.2479e-2, ...; the recorded values on the far coarse cells are 0 or
  +-1 quantum, against AD values of 1e-28..1e-3). On the A3 fixture the
  gradient is carried almost entirely by the 8 tied fine cells per axis
  (|g| up to 992 x / 585 y / 946 z, where AD and central FD agree to
  0.1-0.3 % and AD sits between FD+ and FD- on every tied cell), so the
  non-tied maxima are only 10.0 (x), 0.82 (y) and 0.36 (z), and the
  5 % dominance threshold admits cells whose FD reference is 3-50
  quanta. Every one of the 12 fired cells has |g_fd| <= 17 quanta
  (y: k = 10, 15, 30, 35 at 6-7 quanta; z: k = 12, 15, 16, 17, 27,
  29, 32, 35 at 3-17 quanta); every dominant cell with >= 50 quanta
  (x: 29, 30; y: 16, 29) agrees to <= 1.6 %. AD1 did not meet this
  floor because its dominant cells carry |g_fd| 16-270 (1500-25000
  quanta on 1.4 mm cells). So the fired measurement is consistent with
  a correct joint gradient read against a reference at the f32
  round-off floor of the declared FD, and it is also consistent with a
  joint-gradient error of up to 1.0x on cells that contribute < 1 % of
  the gradient norm — the declared instrument cannot tell these apart,
  and this note does not choose. The joint dx/dy/dz gradient claim is
  therefore NOT supported by this lane; it is not refuted either. A PI
  decision is needed on whether an x64-loss AD3 (the AD1 x64 context,
  which was declared there but not here) is a new declared arm.

- **AD4 HELD**: 20 stacks (nz 6-580, dz_min 1.847 um, dt 6.1e-15 to
  4.5e-13 s, all > 0), forward trace finite and gradient finite on all
  20, zero non-finite values; max |g| per stack from 1.1e-3 (stack 9) to
  7.8e+3 (stack 17), none zero. Wallclock 31.9 s.
- **AD5** (knowledge output): tie tables recorded for AD1 (2 cells) and
  AD3 (8 cells per axis). On every tied cell the AD value lies between
  FD+ and FD- except z k = 20 and k = 26 in AD3 (AD +0.34 / +0.44 vs FD+
  +0.60 / +1.01 and FD- -2.92 / -2.80: inside the interval as well, the
  sign of the central average being the artefact). The realized
  convention is the equal split of the tied `min` cotangent.

Scope statement, recorded as declared: the builder is host-side numpy,
gradients w.r.t. layer EDGES are not available and not claimed; every
AD arm differentiates w.r.t. the realized cell vector (or eps by index).

### Replay test

`tests/unit/nonuniform/test_band_accuracy_ad_replay.py` against the
results JSON: **14 passed, 1 failed** — `test_replay_ad3` red at
`worst dominant AD-vs-FD 5.224e-01 > 0.15`, by design. It stays red; the
answer is a PI decision, not a tolerance edit.

### Validity domain (what this lane measured, and where)

| claim | inside (measured) | outside / not measured |
|---|---|---|
| Stratified-dielectric cavity on a builder MB mesh converges at order 2 with amplitude <= 1.9x uniform (Fabry-Perot / Yee sum k_z^4 d^2 law) | eps 4.3 / 3.0 / 4.3 / 1 (contrast 4.3), 4-cell thin band at cap 1.4, fine 27 and coarse 20 cells per dielectric wavelength in eps 4.3 (s = 1), s = 0.5-2, LSE/Ey m = 1 p = 5 at 10.56 GHz, PEC, dual-cell-average eps on node-aligned interfaces | production interface sampling (first-order, #931, reported not gated); Ez/Hz families; contrast > 4.3; bands thinner than 4 cells; lossy or dispersive media; PML-terminated boxes |
| Absolute accuracy on the same mesh | \|err\| 0.071 % (mb 0.5 dual) to 1.15 % (mb 2 dual) over s; production 0.17-2.05 % | — |
| Auto-z (`_make_dz_profile`) on the same fixture | err 0.070-0.23 % over s, model residual <= 0.031 MHz; no order claimed (thin cell pinned by min_cells_per_feature) | any order statement |
| In-plane two-band grading, TM110 | caps 1.3 and 1.4, 4:1 fine/coarse, 8000 and 12000 steps, 5.79 GHz: \|err\| <= 0.024 %, excess over uniform <= 0.013 pt | caps > 1.4; more than two bands per axis; cavities with dielectric loading in-plane |
| Three axes graded at once, TM111 | cap ~1.3 all axes, 4:1, 7.05 GHz: err -0.041 %, excess -0.039 pt | — |
| Profile gradient (dz), material gradient (eps by index) on a dielectric MB mesh | AD1 1.3e-3, AD2 3.8e-3 at f32, 120 steps, dominant cells with FD reference >= 2000 f32 quanta | cells whose FD reference is < 50 quanta (unresolved by the declared FD) |
| Joint (dx, dy, dz) gradient on a three-axis MB mesh | x axis 4.3e-3 (dominant cells at 49-443 quanta); tied fine cells agree to 0.1-0.3 % on all axes (reported, excluded by AD5) | **y and z dominant coarse/ramp cells: FIRED at 0.52 / 0.99; unresolved — not supported, not refuted** |
| Tracer path of `make_nonuniform_grid` with `cpml_layers = 0` on the builder's output family | 20 stacks, nz 6-580, dz_min 1.85 um: finite, dt > 0 | more than 6 layers; dz below 1.85 um |

Wallclock: A1 482 s of FDTD (18 units; the s = 0.5 units 56-88 s each),
A2 7.8 s, A3 6.6 s, AD1-4 42 s; the whole lane under 12 minutes of
solver time plus 9 s of selfcheck per call.
