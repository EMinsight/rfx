# E2 — thin dielectric layer: resolve it, or average it? — pre-declaration

**Status:** pre-declaration. This commit precedes the instrument
(`validation/research/multiband_nu/e2_thin_layer_subpixel.py`), the replay
test and every measurement. Every numeric window below is frozen from this
commit onward; results are appended under "Results", never edited into the
windows. One attempt per unit; a fired falsifier is a result, not a bug to
tune away.
**Class:** observable-vs-analytic on a z-stratified cavity (family: W7 A1
`w7_accuracy_ad`, note `20260907_nu_band_accuracy_ad_predeclaration.md`,
whose oracle, exact discrete model, harness, extraction and windows are
reused, not re-derived).
**Tree:** worktree `rfx-nu-exp2`, branch `exp/nu-experiments-e2`, based on
`feat/nu-band-accuracy-ad @ fea9f078` (which carries `feat/nu-band-profile`
and `origin/main d990e18c`).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-exp2/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv` and `started_utc` (the W6/W7 convention).
CPU only (a GPU lane runs elsewhere; the machine is shared).
Date: 2026-09-07 (KST).

## 0. Scope — the question and what it costs

The nu-cost lane measured that a thin dielectric layer resolved with 4
cells costs dt x4.7 and a thirds sub-cell x13.9 on the PCB stack (branch
`feat/nu-cost-reduction`, not in this tree). The question here: for a thin
layer of eps 3.0 inside eps 4.3 (the A1 contrast), what is the resonance
error as a function of HOW the layer is represented, at equal transverse
mesh and equal coarse cell:

| arm | representation | eps rule at the nodes |
|---|---|---|
| **R2 / R4 / R8** | the layer resolved with 2 / 4 / 8 cells inside a builder band (`make_band_profile`, cap 1.4, cores free, band protected) | dual-cell average (lane F's gated rule) |
| **S1** | ONE coarse cell holds the layer; that cell's eps is the fill-fraction (volume-weighted) average; the two nodes bounding it take the dual-cell average of the two adjacent cells | subpixel (this lane's rule) |
| **S1b** | as S1 but the layer CENTRED ON A NODE, half in each of two coarse cells (each with fill f/2) | subpixel |
| **S0** | one coarse cell, staircase: the cell's eps is the material at its midpoint (half-open `[lo, hi)`) — the first-order representation, for reference | dual-cell average of the staircased cells |
| **P4** | the R4 mesh with the PRODUCTION column (`Simulation._assemble_materials_nu`, node sampling, lane F section 1) | production |

The "dual" rule of lane F is what every R arm and every reference carries;
S1/S1b replace the cell eps by the fill-fraction average BEFORE the dual
average, which is the only new ingredient. Section 1 derives what it does
and to what order; section 3 freezes the windows.

What is NOT in scope: normal-E (Ez / LSM) families — for them the
fill-fraction rule is the wrong average (section 1c) and a different lane;
loss, dispersion, PML; the Kottke tensor path
(`rfx.geometry.smoothing.compute_smoothed_eps_nonuniform`), which this
lane does not call (its SDF fill fraction is normalised by the geometric
mean of the three local cell sizes, "first-order accurate for non-cubic
Yee cells" per its own docstring, so it would not realise the rule
measured here on a 0.25 x 0.25 x 0.7 mm cell; recorded as the production
subpixel path, not measured); layers thinner than a quarter cell.

## 1. Findings made while writing this note (zero FDTD)

### 1a. Why the arithmetic mean, and what the normal-E rule would be

Ey is tangential to every z-interface of the stack, so Ey is continuous
across an interface and Dy = eps Ey jumps. The Ey update at node k is the
dual-cell (Ampere) balance `d/dt integral_dual(eps Ey) dz = curl H`; with
Ey continuous over the dual cell, `integral(eps Ey) dz = Ey *
integral(eps) dz + O(d^2)`, so the effective eps of a node whose dual cell
contains an interface is the VOLUME average of eps — the arithmetic mean
weighted by the fill fraction. For the normal component (Ez across a
z-interface) Dz is continuous and Ez jumps; the voltage along the edge is
`integral(Ez) dz = Dz * integral(1/eps) dz`, so the effective eps is the
HARMONIC mean `1 / <1/eps>`. That is the Kottke tensor (arithmetic
tangential, harmonic normal) which `rfx.geometry.smoothing` implements
per component; on this fixture only the tangential rule is exercised.

### 1b. To what order the fill-fraction average represents a thin layer, and the law

Continuous problem (lane F 2.1): `psi'' + (eps k0^2 - kx^2) psi = 0`,
Rayleigh quotient `k0^2 = integral(psi'^2 + kx^2 psi^2) / integral(eps
psi^2)`. Replacing eps by another profile `eps + delta` changes the
eigenvalue at first order by `delta(k0^2)/k0^2 = -integral(delta psi^2)
dz / N`, `N = integral(eps psi^2) dz`. For a layer of thickness t and
contrast `Delta = eps_t - eps_c` inside a cell `[z_k, z_k + h]` replaced
by its volume average, `delta(z) = eps_avg - eps_exact(z)` has ZERO
zeroth moment over the cell. Expanding `psi^2` about the cell centre
`z_c`:

`integral(delta psi^2) = psi^2(z_c) * 0 + (psi^2)'(z_c) * M1 + (1/2)(psi^2)''(z_c) * M2 + ...`,
`M1 = integral(delta (z - z_c)) dz = -Delta * t * (z_l - z_c)`,

`z_l` the layer's centroid. So the leading error of the average is the
error of MOVING the layer to the cell centre: `delta f / f = |Delta| * t *
(z_c - z_l) * (psi^2)'(z_c) / (2 N)` (sign for Delta < 0 and (psi^2)' > 0
as here). Two consequences, both measured below:

- at FIXED FILL FRACTION `f = t/h` and the layer at a cell face
  (`z_c - z_l = (h - t)/2`, the worst position inside a cell) the error is
  `K * f (1 - f) * h^2` — **second order in h** — with
  `K = f_true |Delta| (psi^2)'(z_1) / (4 N)`;
- at FIXED PHYSICAL t it is `K * t * (h - t)`: first order in h with
  slope `K t` once h >> t, and zero at h = t (the layer then fills the
  cell exactly and the representation is R1 with dual nodes).

A staircase (S0, one-sided assignment) has a NON-zero zeroth moment
`Delta * (h - t)` or `-Delta * t`, error `f |Delta| psi^2(z_c) (h - t) /
(2 N)` — first order in h at fixed f with a coefficient `psi^2`, not
`(psi^2)'`. Centring the layer on a node (S1b) splits it over two cells
whose first moments cancel by symmetry (`M1 = 0`), leaving the second
moment: higher than second order at fixed f. Production sampling (P4) is
lane F's first-order interface term on the four band cells, sign decided
by the last ulp of the node coordinate (lane F 1a).

Numbers on the E2 reference mode (section 2; `psi` normalised to max 1,
computed on 200 001 z points): `N = 2.600031e-2 m`, `psi(15.4 mm) =
+0.3436`, `(psi^2)'(15.4 mm) = 131.89 /m`, f_ref = 10 680 141 503.244 Hz,
`|Delta| = 1.3`:

**K_pert = 17.608 MHz / mm^2** (the coefficient of `t (h - t)`); staircase
coefficient `f |Delta| psi^2 / (2 N) = 31.53 MHz / mm`.

The exact discrete model (lane F 3.1: the rfx operators with a node eps
column, `sin(w dt/2) = (c0 dt/2) sqrt(lambda)`) gives, for every S1 unit
with t < h, `e_shift` (section 2.4) within **-6.3 % to +9.0 %** of
`K_pert t (h - t)` (table in 2.5); the remainder is the second-moment
term and the "dispersion of the shift" (the coarse mesh's own dispersion
error, -140 MHz at s = 2, scales with the mode frequency, so a +30.6 MHz
layer shift carries about `3 x (30.6 / 10 680) x 140 = 1.2 MHz` of
dispersion, which is why the model's fixed-f S1 order is 1.90 and not
2.00, and why R8 at s = 2, t = 0.7 mm shows -1.19 MHz with the layer
fully resolved).

### 1c. What this means for a normal-E layer (not measured, stated)

For Ez the same expansion applies to `1/eps` (harmonic rule): a
fill-fraction ARITHMETIC average of eps on a normal-E node has a
non-zero zeroth moment in `1/eps` of size `t (1/eps_t - 1/eps_c) -
h (1/eps_avg - 1/eps_c)`, i.e. a first-order error with a `psi^2`
coefficient — the S0 class, not the S1 class. That is why the task's
"arithmetic mean" is right for Ey only, and the lane says so rather than
measuring an Ez family whose interface node carries a discontinuous
component (lane F 2.1 made the same choice).

### 1d. The builder's coarse cells are not exactly dc on the R meshes

`make_band_profile(edges, [dc, t/n, dc, dc], protected = [F, T, F, F],
cap 1.4)` realises the two cores FREE, so their coarse cells shrink to
absorb the ramp: at s = 1, t = 0.35 mm the core-1 cells are 696.784 um
(R2), 686.819 um (R4), 682.701 um (R8) instead of 700, and core 2 is
678.2 / 667.5 / 698.7 um. The air segment is exactly `dc x 20/s` on every
mesh (dc | dc seam, no ramp), so the source and probe planes (2.2) are
nodes of every mesh. "Equal coarse cell" therefore holds to 0.5-2.5 % on
the R arms and exactly on S1 / S1b / S0; the model carries the realised
cells, and the R arms' slightly smaller total errors (2.5) are that.

## 2. Fixture (every dimension an exact node of the uniform coarse lattice at every scale)

### 2.1 Stack and box

| layer | z (mm) | eps_r | note |
|---|---|---|---|
| core 1 | 0 - 15.4 | 4.3 | `15.4 = 11 x 1.4 = 22 x 0.7 = 44 x 0.35` |
| thin | 15.4 - 15.4 + t | 3.0 | t in {0.175, 0.35, 0.7} mm; S1b: 15.4 - t/2 to 15.4 + t/2 |
| core 2 | 15.4 + t - 29.4 | 4.3 | `29.4 = 21 x 1.4` |
| air | 29.4 - 43.4 | 1.0 | `L_z = 43.4 = 31 x 1.4` |

The A1 box otherwise: `a = 30 mm` (x), `b = 3 mm` (y), PEC (`cpml_layers
= 0`), lossless, mu = 1, float32 fields. Scale s in {0.5, 1, 2}: coarse
`dc = 0.7 s mm`, transverse `dx = dy = 0.25 s mm` (A1's proportioning;
the SAME transverse mesh for every arm at a given s). The layer sits at
z = 15.4 mm because the A1 layer's bottom face (14 mm) is a near-null of
this mode (`psi = +0.026`; a layer there would be invisible), while
15.4 mm has `psi = +0.344` and the largest `(psi^2)'` on the coarse
lattice (0.132 /mm): the S1 term of 1b is largest there, and the layer's
own effect (the shift, 2.4) is 6-31 MHz, well above the 0.3 MHz floor.
Reference stack (no layer): core 4.3 over 0-29.4 mm, air above.

Oracle: lane F's transfer matrix (`lse_det`, `lse_roots`, `root_near`)
with these edges; field family LSE / Ey, m = 1, **p = 5**. Frozen
frequencies (Hz):

| stack | f_true | shift vs reference |
|---|---|---|
| reference (no layer) | 10 680 141 503.244 | — |
| t = 0.175 mm at 15.4 | 10 686 187 469.788 | +6.0460 MHz |
| t = 0.35 mm at 15.4 | 10 693 303 832.300 | +13.1623 MHz |
| t = 0.7 mm at 15.4 | 10 710 740 086.538 | +30.5986 MHz |
| S1b t = 0.175 centred at 15.4 | 10 685 646 135.693 | +5.5046 MHz |
| S1b t = 0.35 centred at 15.4 | 10 691 137 566.563 | +10.9961 MHz |
| S1b t = 0.7 centred at 15.4 | 10 702 154 770.606 | +22.0133 MHz |

Same-family neighbours of the t = 0.35 stack: p = 4 at 8.763 GHz
(-18.0 %), p = 6 at 12.306 GHz (+15.1 %). Other families in 9-12.5 GHz
(lane F's second-pass table, recomputed): Ey m = 4 at -0.44 % (source
pair amplitude `sin(4 pi/3) + sin(8 pi/3) = 0`, 8e-16 in float), Hy
(LSM) m = 1 at -0.51 % (not excited by an Ey source). The isolation is
conditional on the source symmetry and polarisation exactly as in lane F;
the harminv line lists are recorded per unit and G3 (0.15 MHz against the
exact discrete model) rules out a mis-identified line.

Oracle self-checks (frozen, must pass before any FDTD): (i) all eps = 1
reproduces `f_mp = (c0/2) sqrt((m/a)^2 + (p/L)^2)` to **<= 1e-12
relative**, m in {1, 5}, a = 30 and 60 mm, 3-16 GHz (scratch: 2.7e-16
over 10 roots); (i') all eps = 4.3, closed form / sqrt(4.3), m = 1, 2-8
GHz (scratch: 3.4e-16, 11 roots); (i'') the Hy determinant, eps = 1,
m = 1 (lane F second pass), <= 1e-12.

### 2.2 Source, probe, waveform, extraction (lane F 2.1 verbatim except the planes)

Ey soft sources at x = a/3 and 2a/3 (nodes: 40/s and 80/s cells), equal
sign, j = ny // 2, both at **z = 35.0 mm** (`= 29.4 + 4 x 1.4`; `psi =
+0.996`, the air antinode); Ey probe at x = a/3, **z = 32.2 mm** (`29.4 +
2 x 1.4`, `psi = +0.799`). Both planes are nodes of every mesh (1d; the
instrument asserts <= 1e-12 m). Gaussian-modulated sine at the unit's
own f_true, sigma_t = 200 ps, t0 = 5 sigma_t; T = **15 ns**; harminv on
the trace after 2 t0 over 7-14 GHz; the mode is the in-band (9.6-11.5
GHz) line nearest f_true, MATCH_GUARD 3 %. Run-length invariance: the
trace truncated to **10 ns** must give a frequency within **0.1 MHz**;
otherwise the unit is INCONCLUSIVE, reported, dropped from every fit and
law. E_FLOOR = 0.3 MHz (fit floor).

### 2.3 Meshes and columns (builder outputs verified on this tree)

- **U** (S1, S1b, S0 and their shared reference): `np.full(124/s...)`:
  nz = 124 / 62 / 31 cells of dc at s = 0.5 / 1 / 2. S1 column: cell
  `k1 = 15.4 mm / dc` (44 / 22 / 11) gets `eps_c + f (eps_t - eps_c)`,
  f = t / dc; S1b: cells k1 - 1 and k1 each get fill f/2; S0: cell
  midpoint rule. Then the dual-cell node average (lane F `dual_eps_nodes`
  applied to the cell vector), float32.
- **R_n** (n = 2, 4, 8): `make_band_profile([0, 15.4, 15.4 + t, 29.4,
  43.4] mm, [dc, t/n, dc, dc], protected = [F, T, F, F], max_ratio =
  1.4)`; verified nz and dz_min (um):

| s | t (mm) | R2 nz / dz_min | R4 nz / dz_min | R8 nz / dz_min | max ratio R2 / R4 / R8 |
|---|---|---|---|---|---|
| 0.5 | 0.175 | 130 / 87.5 | 136 / 43.75 | 143 / 21.875 | 1.319 / 1.344 / 1.361 |
| 1 | 0.175 | — | 78 / 43.75 | — | — / 1.357 / — |
| 1 | 0.35 | 68 / 175 | 74 / 87.5 | 81 / 43.75 | 1.318 / 1.342 / 1.361 |
| 1 | 0.7 | — | 69 / 175 | — | — / 1.318 / — |
| 2 | 0.175 | — | 51 / 43.75 | — | — / 1.364 / — |
| 2 | 0.7 | 37 / 350 | 43 / 175 | 50 / 87.5 | 1.375 / 1.378 / 1.360 |

  R4 at s = 1, t = 0.35 mm (um): `686.819 x 20 | 511.692, 381.22,
  284.015, 211.597, 157.643, 117.447 | 87.5 x 4 | 116.97, 156.364,
  209.027, 279.427, 373.537, 499.342 | 667.518 x 18 | 700.0 x 20`.
  Column: cell midpoint rule (every cell inside one layer) then the dual
  node average. Each R mesh's reference is the no-layer stack on the SAME
  cells.
- **P4**: the R4 mesh with the column `Simulation(boundary = "pec",
  dz_profile = R4, dx)` + three `Box`es `[0, 15.4)`, `[15.4, 15.4 + t)`,
  `[15.4 + t, 29.4)` mm assembles (lane F `a1_production_column`
  generalised); its reference is the no-layer box `[0, 29.4)` on the same
  mesh. Interface nodes as assembled while writing this note (the ulp
  rule, lane F 1a): s = 0.5: 15.4 -> 3.0, 15.575 -> 4.3, 29.4 -> 1.0;
  s = 1: 15.4 -> 3.0, 15.75 -> 4.3, **29.4 -> 4.3** (lower box, on both
  the layer and the reference column); s = 2: **15.4 -> 4.3, 16.1 ->
  4.3** (the layer's face nodes both take the core), 29.4 -> 1.0. The
  instrument re-records these from its own assembled array.

### 2.4 Observables per unit

Per unit: `f_meas` (15 ns), `f_meas_10ns`, the model `f_model`, `err =
f_meas - f_true`, `resid = f_meas - f_model`, invariance, dt, n_steps,
cells, wallclock, the column, the harminv lines. Per (arm, s, t) with its
reference: the **shift error**

`e_shift = (f_meas(layer) - f_meas(ref)) - (f_true(layer) - f_true(ref))`,

the error of the layer's own effect, in which the coarse mesh's
dispersion cancels except for the dispersion-of-the-shift of 1b. Its
model value is the same expression on `f_model`. Everything is defined
from measured numbers; the model is the prediction.

### 2.5 The declared unit set and the model's prediction of every unit

Ladder **L1** (fixed fill f = 1/2, t = dc/2): s = 0.5 / 1 / 2 with t =
0.175 / 0.35 / 0.7 mm; arms S1, S1b, S0, R2, R4, R8, P4 and references
U, R2, R4, R8, P4 — 12 units per scale, 36. Ladder **L2** (fixed t =
0.175 mm): adds S1, S1b, R4 + R4-ref at s = 1 and 2 — 8 units (s = 0.5
shared with L1). Sweep **T** (s = 1, t = dc/4, dc/2, dc): adds S1, S1b,
R4 + R4-ref at t = 0.7 mm — 4 units. **48 units.** Model predictions
(MHz; `err` of the layer unit, `err_ref` of its reference, `e_shift` as
2.4), computed on this tree while writing the note; the instrument's
selfcheck must reproduce every `f_model` to 1e-7 relative:

| unit (arm s t_um) | err | err_ref | e_shift | nz | dt (s) | steps | cells |
|---|---|---|---|---|---|---|---|
| s1 0.5 175 | -8.0883 | -8.6672 | +0.5789 | 124 | 2.8300e-13 | 53004 | 753125 |
| s1b 0.5 175 | -8.6048 | -8.6672 | +0.0624 | 124 | 2.8300e-13 | 53004 | 753125 |
| s0 0.5 175 | -1.4893 | -8.6672 | +7.1779 | 124 | 2.8300e-13 | 53004 | 753125 |
| r2 0.5 175 | -8.1993 | -8.2219 | +0.0226 | 130 | 2.0535e-13 | 73047 | 789275 |
| r4 0.5 175 | -8.0481 | -8.0690 | +0.0209 | 136 | 1.2948e-13 | 115847 | 825425 |
| r8 0.5 175 | -8.1666 | -8.1852 | +0.0186 | 143 | 7.0122e-14 | 213913 | 867600 |
| p4 0.5 175 | +7.6871 | +7.5191 | +0.1680 | 136 | 1.2948e-13 | 115847 | 825425 |
| s1 1 175 | -33.0280 | -34.7443 | +1.7162 | 62 | 5.6600e-13 | 26502 | 99099 |
| s1b 1 175 | -34.5104 | -34.7443 | +0.2339 | 62 | 5.6600e-13 | 26502 | 99099 |
| r4 1 175 | -29.7267 | -29.7712 | +0.0445 | 78 | 1.4024e-13 | 106957 | 124267 |
| s1 1 350 | -32.3940 | -34.7443 | +2.3503 | 62 | 5.6600e-13 | 26502 | 99099 |
| s1b 1 350 | -34.2647 | -34.7443 | +0.4796 | 62 | 5.6600e-13 | 26502 | 99099 |
| s0 1 350 | -16.9394 | -34.7443 | +17.8049 | 62 | 5.6600e-13 | 26502 | 99099 |
| r2 1 350 | -31.2806 | -31.4150 | +0.1344 | 68 | 4.1070e-13 | 36523 | 108537 |
| r4 1 350 | -30.0489 | -30.1509 | +0.1020 | 74 | 2.5896e-13 | 57923 | 117975 |
| r8 1 350 | -31.1871 | -31.2477 | +0.0606 | 81 | 1.4024e-13 | 106957 | 128986 |
| p4 1 350 | -61.0682 | -60.8899 | -0.1784 | 74 | 2.5896e-13 | 57923 | 117975 |
| s1 1 700 | -34.3756 | -34.7443 | +0.3686 | 62 | 5.6600e-13 | 26502 | 99099 |
| s1b 1 700 | -33.8173 | -34.7443 | +0.9270 | 62 | 5.6600e-13 | 26502 | 99099 |
| r4 1 700 | -32.5163 | -32.7204 | +0.2040 | 69 | 4.1070e-13 | 36523 | 110110 |
| s1 2 175 | -136.5783 | -140.1520 | +3.5737 | 31 | 1.1320e-12 | 13251 | 13664 |
| s1b 2 175 | -139.3686 | -140.1520 | +0.7834 | 31 | 1.1320e-12 | 13251 | 13664 |
| r4 2 175 | -104.2638 | -104.1469 | -0.1169 | 51 | 1.4338e-13 | 104616 | 22204 |
| s1 2 700 | -132.0695 | -140.1520 | +8.0825 | 31 | 1.1320e-12 | 13251 | 13664 |
| s1b 2 700 | -136.9034 | -140.1520 | +3.2486 | 31 | 1.1320e-12 | 13251 | 13664 |
| s0 2 700 | -170.7506 | -140.1520 | -30.5986 | 31 | 1.1320e-12 | 13251 | 13664 |
| r2 2 700 | -107.9900 | -108.3166 | +0.3266 | 37 | 8.2139e-13 | 18262 | 16226 |
| r4 2 700 | -98.5541 | -98.5536 | -0.0005 | 43 | 5.1793e-13 | 28962 | 18788 |
| r8 2 700 | -115.0258 | -113.8394 | -1.1865 | 50 | 2.8049e-13 | 53478 | 21777 |
| p4 2 700 | -47.2632 | -39.7986 | -7.4646 | 43 | 5.1793e-13 | 28962 | 18788 |

(The S1 fixed-t point s = 2, t = 0.175 is the `s1 2 175` row; the S0 sign
flips between scales because at f = 1/2 the cell midpoint coincides with
the layer's top face and the last ulp decides — lane F 1a on a cell
instead of a node; at s = 0.5 the midpoint landed inside the layer and the
cell became 3.0, a doubled layer, +7.18 MHz.)

Model orders and ratios (LS slope of log|e| vs log h over h = 0.35 / 0.7 /
1.4 mm):

| quantity | model |
|---|---|
| S1 `e_shift` order, L1 (fixed f) | **1.902** |
| S1b `e_shift` order, L1 | 2.851 |
| S0 `e_shift` order, L1 | 1.046 |
| S1 `e_shift` order, L2 (fixed t = 0.175 mm) | 1.313 (not a power law: `K t (h - t)`) |
| S1 total `err` order, L1 | 2.015; R4 total 1.807 |
| `rho_s = |err_S1| / |err_R4|`, L1 per scale | 1.005 / 1.078 / 1.340 |
| `rho` of the two fitted lines at h1 = 0.7 mm | 1.132 |
| `rho_t` at s = 1, t = 0.175 / 0.35 / 0.7 | 1.111 / 1.078 / 1.057 |
| `|e_shift(S1b)| / |e_shift(S1)|`, L1 per scale | 0.108 / 0.204 / 0.402 |
| S1 `e_shift` vs `K_pert t (h - t)`, the six t < h units | +7.4 / +6.1 / +9.0 / -5.3 / -5.5 / -6.3 % |
| S1 `e_shift / shift_true` at s = 1, t = 0.175 / 0.35 / 0.7 | 28.4 / 17.9 / 1.2 % (R4: 0.74 / 0.77 / 0.67 %) |

### 2.6 The price (dt), declared from the grids

`dt = 0.99 / (c0 sqrt(1/dx^2 + 1/dy^2 + 1/dz_min^2))` (rfx NU CFL); the
transverse 0.25 s mm cells cap the gain, so the z-only ratio
`dc / (t/4)` is quoted beside the realised one:

| s | t (mm) | t / dc | dt(S1) / dt(R4) | z-only | cell-steps R4 / S1 (15 ns) |
|---|---|---|---|---|---|
| 0.5 | 0.175 | 1/2 | 2.186 | 8 | 2.40 |
| 1 | 0.175 | 1/4 | 4.036 | 16 | 5.06 |
| 1 | 0.35 | 1/2 | 2.186 | 8 | 2.60 |
| 1 | 0.7 | 1 | 1.378 | 4 | 1.53 |
| 2 | 0.175 | 1/8 | 7.895 | 32 | 12.83 |
| 2 | 0.7 | 1/2 | 2.186 | 8 | 3.01 |

Wallclock, declared: lane F ran 5.8e10 cell-steps in 88 s on this machine
(6.6e8 /s, loaded); the 48 units total 1.2e12 cell-steps, of which the 12
s = 0.5 units are 1.03e12 (R8 alone 1.9e11 each) — 30-60 min at s = 0.5,
3-6 min at s = 1, seconds at s = 2. Every unit is resumable
(`--arms/--scales/--thicknesses`, merged `--out`).

## 3. Falsifiers (frozen; derivations in section 1; tolerances never widened after measurement)

- **E2-O (oracle):** (i), (i'), (i'') to <= 1e-12 relative (2.1); the
  seven f_true of 2.1 reproduced to <= 1e-2 Hz. Fail => STOP.
- **E2-S (selfcheck, zero FDTD):** every declared mesh regenerated cell
  for cell (1e-12 m), source / probe / interface planes on nodes (1e-12
  m), every `f_model` of 2.5 to 1e-7 relative, every order / ratio of 2.5
  to 1e-3, `K_pert` to 1e-3 relative, the P4 interface tables of 2.3.
  Fail => the instrument refuses to run any arm.
- **E2-G3 (model residual, every one of the 48 units):** `|f_meas -
  f_model| <= 0.15 MHz` (lane F's window; measured there <= 0.06 MHz).
  Fail on a dual / subpixel unit => STOP (the solver does not do what its
  operators say with this column); on a production unit => the column is
  re-read and re-evaluated, still failing => STOP.
- **E2-V:** invariance `<= 0.1 MHz` per unit, else INCONCLUSIVE (dropped
  from every fit, ratio and law; reported).
- **E2-F1 (S1 order at fixed fill):** `p_S1` = LS slope of `log|e_shift|`
  vs `log h` over L1 (s = 0.5 / 1 / 2), points above E_FLOOR 0.3 MHz
  (model 0.579 / 2.350 / 8.083 MHz, all above), in **[1.8, 2.2]** (the
  task's window; model 1.902, the -0.1 being the dispersion-of-shift of
  1b). Fewer than 3 points => INCONCLUSIVE. Fires => the fill-fraction
  node is NOT second order for a tangential field on this solver; record,
  STOP the S1 claim.
- **E2-F2 (S1 costs no more than 4-cell resolving costs over uniform):**
  `rho_s = |err_S1| / |err_R4| <= 2.0` at EACH L1 scale (total resonance
  errors at matched coarse cell and transverse mesh; model 1.005 / 1.078
  / 1.340) and at each T thickness at s = 1 (model 1.111 / 1.078 /
  1.057); the fitted-line `rho` at h1 = 0.7 mm reported (model 1.132).
  2.0 is lane F's A1-F2 window (MB-vs-UC amplitude ratio, model there
  1.883, measured 1.884), taken as the task says. Fires => the subpixel
  cell carries more error than a whole 4-cell band with ramps; record.
- **E2-F3 (the law):** for each of the six S1 units with t < h,
  `|e_shift,meas - K_pert t (h - t)| <= max(0.3 MHz, 0.25 |K_pert t
  (h - t)|)`, `K_pert = 17.608 MHz/mm^2` from the oracle field (1b), no
  fitted constant. Model deviations -6.3 to +9.0 %; the 25 % holds the
  second-moment and dispersion-of-shift terms (11 % at s = 2 in the
  model) plus 0.1 MHz of extraction. Fires => the first-moment law does
  not describe the representation; record which units.
- **E2-F4 (position robustness):** `|e_shift(S1b)| / |e_shift(S1)| <=
  1.0` at each L1 scale (model 0.108 / 0.204 / 0.402): a layer straddling
  a node is represented no worse than a layer at a cell face (the
  worst position of 1b). Fires => the two-cell split is worse than the
  one-cell average; record.
- **E2-R (reported, no gate):** S1b and S0 orders (model 2.851 / 1.046),
  the S1 fixed-t ladder against `K t (h - t)` (model order 1.313 — the
  law, not a power law), R2 / R4 / R8 `e_shift` (model <= 0.33 MHz
  except R8 s = 2 t = 0.7: -1.19 MHz, dispersion-of-shift), P4 `e_shift`
  and total error with its re-recorded interface tables (model L1 +0.168
  / -0.178 / -7.465 MHz; totals +7.69 / -61.07 / -47.26 MHz), the
  thickness sweep `e_shift / shift_true` for S1 vs R4 (model 28.4 / 17.9
  / 1.2 % vs 0.74 / 0.77 / 0.67 %), dt(S1)/dt(R4) and cell-steps per
  (s, t) (2.6), wallclock per unit, the harminv line lists.

## 4. Declared commands, outputs, tests

Instrument `validation/research/multiband_nu/e2_thin_layer_subpixel.py`
(committed BEFORE it is run; imports lane F's oracle, model, extraction
and provenance from `w7_accuracy_ad`; `--arms` from {s1, s1b, s0, r2,
r4, r8, p4, ref}, `--scales`, `--thicknesses` (um), `--ladders` from
{L1, L2, T} selecting the declared unit set, `--out` merged per unit,
`--selfcheck`, `--smoke`, `--force` keeping the earlier row under a
superseded key). Output
`validation/research/multiband_nu/results/e2_thin_layer_subpixel.json`.

```
cd /Users/byungkwankim/Documents/rfx-nu-exp2 && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp2 \
  /Users/byungkwankim/Documents/rfx/.venv/bin/python -c "import rfx; print(rfx.__file__)"
PYTHONPATH=... python -m validation.research.multiband_nu.e2_thin_layer_subpixel --selfcheck --out validation/research/multiband_nu/results/e2_thin_layer_subpixel.json
PYTHONPATH=... python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1,L2,T --scales 2 --out ...
PYTHONPATH=... python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1,L2,T --scales 1 --out ...
PYTHONPATH=... python -m validation.research.multiband_nu.e2_thin_layer_subpixel --ladders L1 --scales 0.5 --out ...   (split per arm as the 600 s cap requires)
```

Order: selfcheck -> s = 2 -> s = 1 -> s = 0.5 -> results commit(s). One
attempt per unit; if an instrument defect forces a re-run, both rows stay
in the JSON and the note says so.

Replay test `tests/unit/nonuniform/test_e2_thin_layer_subpixel_replay.py`
(no FDTD): pins the frozen windows against the instrument's constants;
re-runs the oracle checks; recomputes `K_pert` and the S1 L1 model ladder
(order 1.902 to 1e-3) from first principles; with the JSON present,
regenerates every mesh (1e-12 m), recomputes every `f_model` from the
JSON's own profile and column (1e-9 relative), re-evaluates every rule of
section 3 through the instrument's judge and must agree with the verdict
the JSON carries. A rule that FIRED turns its replay test red by design.

## 5. Impact-sweep rule and the #931 note

This lane adds an instrument, a results JSON, a test and this note; it
changes NO `rfx/` source. The subpixel column is assembled by the
instrument and handed to `run_nonuniform` as a node-wise
`MaterialArrays.eps_r`, exactly as lane F's dual rule was. #931
(geometry -> lattice ownership) gets a third record: the fill-fraction
average with dual nodes is the second-order representation of a thin
tangential-E layer at fixed fill; at fixed physical thickness its error
is `K t (h - t)` — the layer is effectively moved to the cell centre —
and the production node sampling stays first order with an ulp-decided
sign (2.3). What the production path should do with a sub-cell layer is
#931's question; this lane measures both and quotes the law.

## 6. Where this note departs from the task as written, and why

- The layer sits at 15.4 mm, not at the A1 layer's 14 mm (a near-null of
  this mode; the representation error would be invisible there); the air
  interface moves to 29.4 mm and L_z to 43.4 mm so every plane is a node
  of the uniform coarse lattice at every scale.
- "S1 fitted order" is fitted on the SHIFT error (2.4), not the total
  error: the total is 90-95 % coarse-mesh dispersion and would fit 2.0
  for any representation (model: S1 total 2.015 and S0 total 3.42, a
  meaningless number); the total-error ratio is what E2-F2 uses, as the
  task's "matched coarse cell" comparison.
- Two ladders (fixed fill, fixed thickness) instead of one, because the
  derivation of 1b says the order differs between them and both are what
  a user meets.
- S0 (staircase) added as a reported reference arm on the same mesh; the
  task's R2 / R4 / R8 ordering is read as "cells in the layer" at equal
  coarse cell (the A1 MB s = 2 / 1 / 0.5 realisations hold 2 / 4 / 8
  cells).
- S1b at fixed fill has fill f/2 = 1/4 per cell; the task's "fill
  fraction 0.5 each side" is the T-sweep unit `s1b 1 700` (t = dc split
  over two cells), which is also run.
- The builder's free cores are 0.5-2.5 % smaller than dc on the R meshes
  (1d); reported, the model carries the realised cells.
