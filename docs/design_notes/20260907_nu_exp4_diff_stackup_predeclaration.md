# E4 — differentiable stackup prototype (thickness and permittivity as design variables) — pre-declaration

**Status:** pre-declaration. This commit precedes the instrument
(`validation/research/multiband_nu/e4_diff_stackup.py`), the fast test and
every measurement. Every numeric window below is frozen from this commit
onward; results are appended under "Results", never edited into the
windows. One attempt per arm; a fired falsifier is a result, not a bug to
tune away.
**Class:** autodiff integrity of a research-level stackup map (family: W5
`w5_ad_consistency`, W7 AD1/AD2 `w7_accuracy_ad`,
`tests/unit/autodiff/test_nonuniform_forward_grad.py`). A research script;
NO production (`rfx/`) change.
**Tree:** worktree `rfx-nu-exp4`, branch `exp/nu-experiments-e4`, based on
`feat/nu-band-accuracy-ad @ fea9f078` (which carries `feat/nu-band-profile`
and `origin/main d990e18c`).
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-exp4/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `git_modified_tracked`, `argv` and `started_utc` (the W6/W7
convention).
Date: 2026-09-07 (KST).

## 0. Scope and the question

What this lane sits on (cited, not re-derived): on the A1 stack of the W7
note (core 4.3 | thin 3.0 | core 4.3 | air, interfaces on nodes) the profile
gradient `d L / d dz` (AD1, worst 1.31e-3 vs central FD at f32, 120 steps)
and the material gradient `d L / d eps_thin` with eps attached by node index
(AD2, 3.8e-3) HELD (`results/w7_accuracy_ad.json`, note
`20260907_nu_band_accuracy_ad_predeclaration.md` sections 2.4-2.5, 3.5).
Both differentiate w.r.t. the REALIZED cell vector or a scalar set on a
fixed node set; neither differentiates w.r.t. a physical design variable of
the stack.

The question this lane measures: can a substrate THICKNESS and a layer
PERMITTIVITY be design variables with correct gradients TODAY, on the
committed NU solver, through

- (a) mesh stretching with fixed topology — the thin layer keeps a fixed
  cell count and its cells scale with the thickness; the neighbouring
  runs absorb the change so the column length is fixed; every interface
  stays on a node;
- (b) a fill-fraction eps on the interface cell — the interface NODE
  takes the dual-cell-average eps, which is the volume fill fraction of
  the dual cell centred on that node and is differentiable in the layer
  eps AND, through the two adjacent cell sizes, in the thickness.

"Correct" means: `jax.grad` agrees with central finite differences in the
design variable itself, within the committed NU AD convention (15 %, sign
agreement), on two observables — the W7 AD1 probe-energy loss and a
narrowband DFT power at the analytic resonance of the nominal stack — and
the sign of the resonance-proxy gradient agrees with the direction the
analytic resonance moves (transfer-matrix oracle of W7).

**Scope statement, recorded so nobody reads more into this than it says.**
Coordinate-based rasterization is NOT a usable differentiation path for a
stackup: `Box.mask_on_coords` is traceable but its volume branch is a
boolean comparison of node coordinates against the faces
(`(coords >= lo) & (coords < hi)`) and its thin branch an `argmin` — the
realized eps is piecewise constant in every coordinate and its derivative
w.r.t. a cell size is identically zero wherever it exists (section 1c
measures exactly that: finite, `max |g| = 0`). This prototype therefore
attaches materials BY NODE INDEX on a topology that never changes with the
design variables, and carries the interface fill fraction as an explicit
differentiable weight. That is the design constraint for any production
version; this lane records it, it does not implement one.

## 1. Findings made while writing this note (zero FDTD)

### 1a. The E4 stack IS the A1 stack; the oracle's slopes

The fixture (section 2) is the A1 stack at s = 1 on a 0.5 mm transverse
cell. The transfer-matrix oracle of W7 (`lse_det`, LSE / Ey family, kx =
pi / 30 mm) with edges `[0, 15 - h/2, 15 + h/2, 30, 44] mm` and eps
`[eps_core, eps_thin, eps_core, 1]` at the nominal `(h, eps_thin,
eps_core) = (2 mm, 3.0, 4.3)` returns `f_res = 10 561 719 600.896 Hz`,
bit-identical to W7's `f_true` (the same root of the same determinant).
Central differences of that root (float64, the FD steps of section 3):

| parameter | step | `d f_res / d p` | `f_res` moves per step | direction |
|---|---|---|---|---|
| h_thin | 1e-3 relative (2 um) | **+2.2275e10 Hz/m** | +44.6 kHz (+4.2e-6) | up |
| eps_thin | 1e-2 relative (0.03) | **-2.3779e7 Hz** | -713 kHz (-6.8e-5) | down |
| eps_core | 1e-2 relative (0.043) | **-8.3599e8 Hz** | -35.9 MHz (-3.4e-3) | down |
| eps_thin | 1e-3 relative (0.003) | -2.378e7 Hz | -71 kHz | down |
| eps_core | 1e-3 relative (0.0043) | -8.360e8 Hz | -3.6 MHz | down |

The signs are the physics: a thicker eps 3.0 layer replacing eps 4.3 lowers
the mean eps of the column and raises f; raising either eps lowers f. The
eps_core row at 1e-2 is why section 2.4 uses 1e-3 on the DFT observable:
at 1e-2 one FD step moves the line by 55 % of the flank offset Delta
(65.6 MHz) and the central difference is no longer in the linear regime of
a sinc^2 line — a reference defect, not a gradient defect.

### 1b. The NU Kottke smoother at a node-aligned dielectric | dielectric interface

`rfx.geometry.smoothing.compute_smoothed_eps_nonuniform` (read in full
before this claim) is the only fill-fraction path on the NU grid. On the
nominal E4 mesh with the three A1 boxes (`[0,14)`, `[14,16)`, `[16,30)` mm,
eps 4.3 / 3.0 / 4.3, background 1.0) its Ey component at the central
(i, j) reads, at the three interface nodes:

| node | z (mm) | adjacent cells (mm) | Kottke `eps_ey` | dual-cell average | arithmetic mean |
|---|---|---|---|---|---|
| 20 | 14.0 | 0.7 \| 0.5 | **2.8250** | 3.7583 | 3.65 |
| 24 | 16.0 | 0.5 \| 0.7 | **2.8250** | 3.7583 | 3.65 |
| 44 | 30.0 | 0.7 \| 0.7 | 2.6500 | 2.6500 | 2.65 |

Mechanism, from the code: shapes are grouped by eps and the groups are
applied in order, each blended against the field ASSEMBLED SO FAR. At
node 20 the eps 4.3 group (both cores, SDF union = 0 on the face, fill
f = 0.5) is first averaged against the BACKGROUND air, `(4.3 + 1)/2 =
2.65`, and the eps 3.0 group is then averaged against that, `(3.0 +
2.65)/2 = 2.825` — a value below BOTH materials that meet there. The fill
fraction itself is `clip(0.5 - sdf / (dx dy dz)^(1/3))`, which at a node
ON the face is 0.5 regardless of the two adjacent cell sizes, so even with
the grouping fixed the smoother would give the arithmetic mean (3.65), not
the dual-cell average (3.758) that is the second-order-consistent value
for a tangential component between unequal cells (W7 note 1b). The
smoother is therefore not usable as path (b) for a multilayer stack today;
this lane assembles the dual-cell average itself. Recorded for #931, not
changed (section 5).

### 1c. `Box.mask_on_coords` under a tracer: gradient finite and identically zero

`jax.grad` of `sum(where(Box([14,16] mm).mask_on_coords(0, 0, cumsum(dz)),
3.0, 4.3) * k)` w.r.t. the 64-cell nominal dz vector: no exception, every
entry finite, `max |g| = 0.0`. The traced path exists (GEO Tier-2 / #802)
but carries no derivative — the scope statement of section 0.

## 2. Fixture and the stackup map (every quantity declared; the instrument asserts it in `--selfcheck`)

### 2.1 Box, mesh, materials

The A1 box of W7 note 2.1 at s = 1: `L_z = 44 mm`, `a = 30 mm` (x),
`b = 3 mm` (y), PEC-closed (`cpml_layers = 0`, `run_nonuniform` on
`make_nonuniform_grid(domain_xy=(30, 3) mm, dz_profile, dx=0.5 mm)`),
lossless, mu = 1, float32 fields. Transverse cell `dx = dy = 0.5 mm` (=
df, NOT A1's df/2 = 0.25 mm: half the transverse nodes for the reverse-mode
tape of the 8000-step arm; e_x of the m = 1 mode is second order in dx and
irrelevant to a gradient-consistency check). Grid `(nx, ny, nz) = (61, 7,
65)` = 27 755 nodes. Nominal dz (the W7 A1 MB s = 1 vector, 64 cells):
`[0.7 mm] x 20 | [0.5 mm] x 4 | [0.7 mm] x 20 | [0.7 mm] x 20`; interfaces
at nodes k = 20 (14 mm), 24 (16 mm), 44 (30 mm); the thin layer has FOUR
cells, the cap-1.4 seam on both sides. `dt = 0.99 / (c0 sqrt(1/dx^2 +
1/dy^2 + 1/dz_min^2)) = 9.5329e-13 s` at the nominal (float64; the f32
grid value is recorded).

### 2.2 The map `stackup(params) -> (dz_profile, eps_by_node)`

`params = (h_thin, eps_thin, eps_core)`, nominal `(2e-3 m, 3.0, 4.3)`, a
length-3 `jnp` array (float32 under the default x64 = 0, as every W7 AD
arm). With `N_CORE = 20`, `N_THIN = 4`, `N_AIR = 20`, `T_CORE = 14 mm`,
`H0 = 2 mm`, `T_AIR = 14 mm`:

- `d_thin = h_thin / 4` (four identical cells; the thin layer stays exactly
  `[15 - h/2, 15 + h/2] mm`);
- `d_core = (T_CORE - (h_thin - H0) / 2) / 20 = (15 mm - h_thin / 2) / 20`
  on BOTH cores (the two neighbouring free runs absorb half the change
  each: the thin layer is centred at 15 mm, the core | air interface stays
  at 30 mm, the column stays 44 mm);
- `d_air = T_AIR / 20 = 0.7 mm`, fixed;
- `dz = [d_core] x 20 ++ [d_thin] x 4 ++ [d_core] x 20 ++ [d_air] x 20`.

Every cell is linear in h_thin, so the map is smooth (C-infinity) in
h_thin, the topology (64 cells, 65 nodes, interface nodes 20 / 24 / 44)
never changes, and the Jacobian `d dz / d h_thin` is `-1/40` on the 40
core cells, `+1/4` on the 4 thin cells, `0` on the 20 air cells (sum 0:
the column length is invariant). Column length `44 mm` to <= 1e-9 m at
every h the instrument evaluates (asserted).

Cell eps: `[eps_core] x 20 ++ [eps_thin] x 4 ++ [eps_core] x 20 ++ [1.0] x
20`. Node eps (65 nodes; node 64 is the bounding / PEC node):
`eps_node[0] = eps_cell[0]`, `eps_node[64] = eps_cell[63]`, and for
`1 <= k <= 63`

    eps_node[k] = (eps_cell[k-1] dz[k-1] + eps_cell[k] dz[k]) / (dz[k-1] + dz[k])

— the dual-cell average of W7's "dual" rule, i.e. the fill fraction of the
dual cell centred on node k. It is the identity away from interfaces and
at the three interface nodes it depends on h_thin through the two cell
sizes (path (b)): at the nominal, node 20 and 24 carry `(4.3 x 0.7 + 3.0 x
0.5) / 1.2 = 3.7583`, node 44 carries `2.65`. The eps array handed to the
solver is `broadcast(eps_node[None, None, :])`, a `MaterialArrays.eps_r`
of the grid shape, mu = 1, sigma = 0 — exactly the AD1/AD2 mechanism with
the column now a function of params.

### 2.3 Observable L1 — the W7 AD1 probe-energy loss

Sources: Ey soft sources at the A1 pair `x = a/3, 2a/3` (i = 20, 40),
`j = 3` (= ny // 2), `k = 24` (the thin | core-2 interface node, the
antinode of the target mode), equal sign, the W5 waveform by step index
`exp(-((n - 15) / 5)^2)`; probe Ey at (20, 3, 20) (the core-1 | thin
interface node, across the thin band from the source, the AD1 layout);
`N1 = 120` steps (114 ps at the nominal dt: the pulse crosses the band,
~11 ps, and reflects within the cores); `L1 = sum(ts^2)`. Materials,
sources and probe are attached by node index, so with h_thin the source
and probe planes move physically (`z = 15 +- h/2`): that is the map, and
the derivative includes it.

### 2.4 Observable L2 — narrowband DFT power at the analytic resonance

Same box, sources and pair; the drive is A1's Gaussian-modulated sine at
`f_res` (section 1a), `exp(-((t - t0) / sigma_t)^2) sin(2 pi f_res t)`,
`sigma_t = 200 ps`, `t0 = 5 sigma_t = 1 ns`, `t = n dt` (dt traced: the
drive follows the mesh's own dt, as the FD sees it); probe Ey at
(20, 3, 44) (the core-2 | air interface node, |psi| = 0.975 for the target
mode; the AD1 probe node 20 is a near-null, |psi| = 0.033, and is not used
here). `N2 = 8000` steps (7.626 ns at the nominal dt), `run_nonuniform(...,
checkpoint_every=100)` (scan-of-scan remat; the forward values are those of
the plain scan). The DFT of the probe trace over the whole run,

    X(f) = dt * sum_n ts[n] exp(-i 2 pi f n dt),     P(f) = |X(f)|^2,

evaluated at three frequencies from one run:

- `P_nom = P(f_res)`, the task's observable (gated for AD vs FD);
- `P_up = P(f_c + Delta)`, `P_dn = P(f_c - Delta)` with `Delta = 0.5 / (N2
  dt_nominal) = 65.56 MHz` (the steep part of the sinc^2 main lobe, whose
  first null is at 1 / T = 131.1 MHz) and `f_c` the frequency of the
  DISCRETE line on the nominal mesh — W7's exact discrete stratified model
  (`discrete_lambda` on this dz with the dual eps column, `mu_x` of the
  60-cell 0.5 mm x mesh, `leap` at the nominal dt), computed and recorded by
  `--selfcheck` (W7 measured the same stack at dx = 0.25 mm at
  f_true - 30.1 MHz; here e_x differs slightly and the selfcheck says by how
  much). Centring the flanks on the discrete line, not on `f_res`, keeps
  both flank points inside the main lobe and off its peak by a declared
  margin (30 +- 66 MHz against a 131 MHz null);
- `S = (P_up - P_dn) / (P_up + P_dn)`, the flank asymmetry: the smooth
  proxy for "how far the line moved", which cancels the excitation
  amplitude to first order. `S` increases when the line moves UP in
  frequency.

`f_res`, `f_c`, `Delta`, the three evaluation frequencies and `dt_nominal`
are float64 constants fixed at selfcheck time; only the trace and dt inside
the loss are traced.

Why the flanks and not `P_nom` alone for the physics sign: at the line
centre `d P / d f_res = 0` to first order, so the sign of `d P_nom / d h`
is decided by the excitation amplitude's dependence on h (source node
eps, source-plane motion), not by the direction the line moves; it is
not a physics prediction. The task's literal line (sign of `d P_nom / d
h_thin` vs the oracle) is REPORTED with that caveat; the gated physics
sanity (E4-S) is on `S`.

### 2.5 Gradients, FD references, the tie

`jax.jacrev` of `(P_nom, P_up, P_dn)` w.r.t. params (three reverse passes
on one forward) gives `d P_nom / d p` and, by the quotient rule on the
jacobian rows, `d S / d p`; `jax.grad` of L1 w.r.t. params. Both on the f32
path (`jax.jit`).

Central FD in the DESIGN VARIABLE, `g_fd = (L(p + h) - L(p - h)) / 2h`,
also `FD+ = (L(p + h) - L(p)) / h` and `FD- = (L(p) - L(p - h)) / h`:

- L1: `h = 1e-3 h_thin` (2 um), `h = 1e-2 eps_thin` (0.03), `h = 1e-2
  eps_core` (0.043) — the task's steps;
- L2 (`P_nom`, `S`): `h = 1e-3` relative on all three (section 1a: at 1e-2
  the eps_core step moves the line by 55 % of Delta); the 1e-2 rows for
  eps_thin / eps_core on L2 are RECORDED (two extra forwards each) and
  reported, not gated.

Resolution floor (the W7 second-pass rule, declared here before the run):
`quanta = |L(p + h) - L(p - h)| / ulp_f32(L(p))`; a reference with `quanta <
50` is UNRESOLVED and that (loss, parameter) row is INCONCLUSIVE — reported
with its AD, FD+, FD- and quanta, not fired. Expected from section 1a and
the line shape: h_thin on L2 ~1e4 quanta, eps_core on L2 ~1e5, L1 rows
>= 1e3 (the AD1 dominant cells sat at 1500-25000 quanta).

**The min-tie, declared handling.** `make_nonuniform_grid` sets dt from
`jnp.min` of the padded dz; the four thin cells tie for that minimum
exactly (same f32 expression `h_thin / 4`). No softmin, no jitter, no
temperature: the design variable h_thin moves ALL FOUR tied cells together,
so `dz_min = h_thin / 4` is linear in h_thin, `d dz_min / d h_thin = 1/4`,
and JAX's realized convention on a tied `min` (W7 AD5: the cotangent split
equally over the tied entries) composes with the Jacobian to exactly `4 x
(1/4) x (1/4) = 1/4` — the equal split is EXACT when the tied cells move as
one. The FD in h_thin is smooth for the same reason; a kink would show as
`FD+ != FD-` at O(1). Recorded, not gated, alongside (the AD5 knowledge
output): the per-cell gradient `g_dz = d L1 / d dz` at the nominal (the
same loss written on the dz vector with the cell eps fixed by index and
the node eps re-assembled from dz), the one-sided per-cell `FD+` / `FD-`
of L1 for each of the four tied cells (`h = 1e-3 dz[k]`, one cell at a
time — the case that DOES suffer the tie), the equal-split model `FD+ +
(FD- - FD+) / 4`, and the chain-rule composition `J^T g_dz` against the
direct `d L1 / d h_thin` (the same value up to f32 summation order;
reported).

### 2.6 Cost, declared

W7 measured 2.9e8 cell-steps/s on this machine for a 19 k-cell A1 unit;
assume half while the machine is shared. L1: 27 755 x 120 = 3.3e6
cell-steps, negligible; L2: 2.2e8 per forward (~1.5 s), 3 reverse passes
with remat (~3 forwards each) ~15 s, 10 forwards for FD (6 at 1e-3, 4 at
1e-2) ~15 s, the L2 arm under 2 minutes including compile. Reverse-mode
tape at `checkpoint_every = 100`: 80 segment carries + 100 inner steps of
0.67 MB each, ~120 MB.

## 3. Falsifiers (frozen; tolerances never widened after measurement)

All comparisons on the f32 path; "rel" = `|g_ad - g_fd| / |g_fd|`;
"sign" = `sign(g_ad) == sign(g_fd)`. A row whose reference is unresolved
(quanta < 50) is INCONCLUSIVE, reported, not counted either way.

- **E4-F1 (L1, mesh-and-material gradient of the probe energy):** for
  each of h_thin, eps_thin, eps_core: sign agreement and **rel <= 0.15**
  (the committed NU AD convention: `test_grad_wrt_dz_profile_matches_fd`,
  W5, W7 AD1). Fires on any parameter => the design-variable gradient is
  NOT supported for that parameter on this observable; record, no re-run.
- **E4-F2 (L2 = `P_nom`, the resonance-proxy gradient):** the same rule,
  same three parameters, FD step 1e-3 relative. Fires => as above on the
  narrowband observable.
- **E4-S (physics sanity, gated):** `sign(d S / d p)` from AD equals the
  sign of `d f_res / d p` from the transfer-matrix oracle (section 1a) for
  p = h_thin (**+**), eps_thin (**-**), eps_core (**-**); the FD sign of S
  is recorded beside it. Fires => either the line-shape proxy is not
  reading the resonance (reported with `P_up`, `P_dn` and the recorded
  quanta) or the solver's discrete line moves against the physics; the
  note says which from the recorded numbers, it does not re-run.
  Reported with it, not gated: the oracle-and-sinc^2 prediction of
  `d S / d p` (`(d S / d delta)|_0 x d f_res / d p`, delta the line offset)
  and the ratio AD / prediction; the exact discrete model's own
  `d f_c / d p` (float64 central FD on `discrete_lambda`) and its ratio to
  the oracle's; the literal `sign(d P_nom / d h_thin)` vs `+`.
- **E4-T (smoothness in h_thin — the tie does not bite):** on L1 and on
  `P_nom`, `|FD+ - FD-| / |g_fd| <= 0.15` for h_thin. Fires => the loss has
  a kink in the design variable (the tied minimum or something else),
  reported with the tie table. The tie table itself (section 2.5) is a
  knowledge output, no gate.
- **E4-AD (tracer path):** any `TracerArrayConversionError` or exception
  on the traced path (`cpml_layers = 0`, `checkpoint_every = 100`,
  `jacrev`) is a fired AD witness: recorded, the arm STOPs, no re-run with
  a different setting.
- **E4-M (map invariants, `--selfcheck`, no FDTD; a failure is an
  instrument defect and STOPs everything):** column length 44 mm to
  <= 1e-9 m at h_thin in {1.8, 2.0, 2.2} mm; the four thin cells equal to
  the f32 bit; interface nodes at k = 20 / 24 / 44 with `|z - (15 - h/2,
  15 + h/2, 30) mm| <= 1e-9`; `jax.jacfwd` of dz w.r.t. h_thin equal to the
  declared `[-1/40 x 20, +1/4 x 4, -1/40 x 20, 0 x 20]` to 1e-6 relative;
  node eps at 20 / 24 / 44 equal to the hand formula to 1e-6; the oracle
  self-checks (i)/(i') of W7 re-run (<= 1e-12 relative); `f_res` of the
  nominal equal to W7's `F_TRUE_DECLARED` to 1e-9 relative; the three
  oracle slopes of section 1a reproduced to 1e-4 relative; the section 1b
  Kottke column reproduced (2.825 / 2.825 / 2.65 to 1e-6); section 1c
  reproduced (`max |g| = 0`).

## 4. Declared commands, outputs, tests

Instrument `validation/research/multiband_nu/e4_diff_stackup.py`
(committed BEFORE it is run). `--selfcheck` runs section 3 E4-M and
refuses to run any arm if one fails; `--arms l1,l2`; `--smoke` (L1 at 12
steps, L2 at 200 steps, FD on h_thin only, written under the key
`"smoke"`, never judged) is the one permitted bring-up call before the
measurement; `--out` merged per arm. Output
`validation/research/multiband_nu/results/e4_diff_stackup.json`: per arm
the provenance block, the nominal dz and eps columns, dt (float64 and the
grid's f32 value), every loss value used, every AD / FD / FD+ / FD- value,
quanta, every gate value and verdict, the tie table, wallclock.

```
cd /Users/byungkwankim/Documents/rfx-nu-exp4 && PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp4 \
  /Users/byungkwankim/Documents/rfx/.venv/bin/python -c "import rfx; print(rfx.__file__)"
PYTHONPATH=... python -m validation.research.multiband_nu.e4_diff_stackup --selfcheck --out validation/research/multiband_nu/results/e4_diff_stackup.json
PYTHONPATH=... python -m validation.research.multiband_nu.e4_diff_stackup --smoke --out validation/research/multiband_nu/results/e4_diff_stackup.json
PYTHONPATH=... python -m validation.research.multiband_nu.e4_diff_stackup --arms l1 --out validation/research/multiband_nu/results/e4_diff_stackup.json
PYTHONPATH=... python -m validation.research.multiband_nu.e4_diff_stackup --arms l2 --out validation/research/multiband_nu/results/e4_diff_stackup.json
```

Order: selfcheck -> smoke -> L1 -> L2 -> results commit. One attempt per
arm. If the smoke reveals an instrument defect (an exception, a shape
error), the fix is a commit BEFORE the measurement and is listed in
Results; a smoke that runs is not looked at for anything but "it runs".

Fast test `tests/unit/nonuniform/test_e4_diff_stackup.py` (<= 30 s,
tolerances written before its first run): (a) the E4-M map invariants
without FDTD (column length, four equal thin cells, interface nodes, the
Jacobian, the node eps formula); (b) one live gradient check of L1 at
`N_STEPS = 40` on the (6 mm, 3 mm) transverse box (source pair at i = 4,
8 of 12 cells, the same k's): finite gradient; h_thin and eps_thin
resolved (quanta >= 50, asserted) and within 15 % of central FD with sign
agreement; eps_core within 15 % with sign IF resolved, else its quanta are
printed and it is not compared. Same steps as section 2.5.

## 5. Impact-sweep rule and the #931 records

This lane adds a research script, a results JSON, a test and this note.
It changes NO `rfx/` source. If bring-up finds an `rfx/` defect the fix
is a separate commit before any measurement, with the band-profile note's
F6 rule applied verbatim (every locked value that moves listed old -> new
with a physical justification; an unjustifiable move => STOP).

Two records for #931 (geometry -> lattice ownership), neither changed here:

1. The only fill-fraction path on the NU grid,
   `compute_smoothed_eps_nonuniform`, blends each eps group against the
   previously assembled field and normalizes the fill fraction by the
   geometric mean of the three cell sizes; at a node-aligned dielectric |
   dielectric interface between unequal cells it returns 2.825 where the
   two materials are 4.3 and 3.0 and the second-order value is 3.758
   (section 1b). A differentiable stackup on the production path needs
   the dual-cell fill fraction on the interface node, per component,
   with the cell sizes as its weights.
2. Coordinate rasterization (`Box.mask_on_coords`) is traceable with
   identically zero derivative (section 1c); materials of a
   differentiable stack must be attached by node index on a fixed
   topology, with the interface weight carried separately (section 0).

## 6. Where this note departs from the task as written, and why

- L2's FD step is 1e-3 relative on eps_thin / eps_core, not the task's
  1e-2 (section 1a: at 1e-2 the eps_core step moves the resonance by 55 %
  of the flank offset; the reference would be nonlinear, not the
  gradient wrong). The 1e-2 rows are recorded and reported. L1 keeps the
  task's steps.
- The gated physics sign is on the flank asymmetry `S`, not on `P_nom`
  (section 2.4: at the line centre the sign of `d P / d h` is not a
  physics prediction). The literal `P_nom` sign is reported.
- The flanks are centred on the discrete line `f_c` (exact discrete
  model), `P_nom` on the analytic `f_res` as the task says.
- Both cores absorb the thickness change (the thin layer stays centred
  at 15 mm); the air run is fixed.
- Transverse cell 0.5 mm, not A1's 0.25 mm (tape size of the 8000-step
  arm; a gradient-consistency check, not an accuracy ladder).
- The tie is handled by the design variable moving all four tied cells
  together (the task's second option), with the per-cell one-sided table
  recorded; no softmin.
