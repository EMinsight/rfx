# G4 — CPML localization — pre-declaration

**Status:** pre-declaration, before code changes or validation runs. Gates
below are frozen by this commit. Results will be appended, never used to
rewrite a gate.
**Tree:** `/Users/byungkwankim/Documents/rfx-nu-cost`,
`feat/nu-cost-reduction`, baseline `c30d30208ff3fc3b73cf7d24a2f0fcf6fd08025b`.
Date: 2026-09-10 (KST).

## Mechanism found by reading

Read the G1b measured section and its committed JSON first. At 300^3,
CPML-8 costs 4.722608 plain steps on bare-slow and 5.636902 on nu-uniform.
The L=4/16 costs are 0.492497/0.490897 ns and 0.557011/0.645493 ns,
while absorbing fraction grows 0.075676 -> 0.261472. This supports a
whole-array cost in the CPML step, but does not identify its operator.

The proposed profile-expansion mechanism is **not what this HEAD does**.
Line references below are to baseline `rfx/boundaries/cpml.py`:

- 570: `return arr if depth >= n_alloc else arr[:depth]`;
  580: `return arr if depth >= n_alloc else arr[n_alloc - depth:]`.
  Neither helper expands a profile to the domain.
- 668: `b_x_lo = _clip_lo(px_lo.b, n_x, n)[:, None, None]` is
  `(n_x, 1, 1)`, broadcasting over a slab, not over the whole domain.
- 731: `new_psi_ey_xlo = b_x_lo * cpml_state.psi_ey_xlo + c_x_lo * curl_hz_dx_xlo`;
  732: `ey = ey.at[:n_x, :, :].add(-ce_xlo * new_psi_ey_xlo)`.
  Psi arithmetic and correction operands are already slab-sized. Psi
  allocation at 541–554 uses `(axis_depth, perpendicular_1, perpendicular_2)`.
- 637: `_ce_full = dt / (materials.eps_r * EPS_0)`;
  911: `_ch_full = dt / (materials.mu_r * MU_0)` compute full coefficients
  before slicing. 728: `hz_shifted_xlo = _shift_bwd(state.hz, 0)[:n_x, :, :]`
  shifts a full field before slicing. `_shift_bwd/fwd` in `core/yee.py`
  pad and slice full arrays. Whether XLA eliminates this excess is unknown.
- Each `.at[slab].add` returns a full field array. There are two ordered
  adds per face/component (psi then kappa), including at intersecting faces.
  This is a candidate source of full-volume fusion/scatter traffic.

Uniform setup stores psi in `carry_init["cpml"]` (simulation.py:844–845).
Its core step passes carry through H CPML (1302), then E CPML (1399),
and into the next scan iteration. NU does the same (nonuniform.py:2044,
2087), using `cpml_axes_eff` selected from allocated pads (1874).
Both call the same CPML functions with materials. NU initialization passes
PEC/PMC sets explicitly and retains independent z-lo/z-hi cell sizes.
The existing active-depth clamp has legacy-grid and excluded-axis guards;
these semantics must not be inferred solely from zero padding.

## Intended change

Localize remaining coefficient and shifted-neighbor expressions before
arithmetic. Replace scatter-add field corrections with slab read/add/write
using a contiguous dynamic-update-slice, keeping each addition separate and
in the existing order. Further restrict a face to its own active region
only where existing profile metadata proves the omitted region is no-op;
retain all psi carry shapes and untouched padding. No profile, cell size,
boundary token, or caller policy change. Both lanes share this implementation.
Compiler fusion may still reorder arithmetic despite identical Python
parenthesization: that is a possible gate failure, not permission to relax it.

## Frozen gates

1. **BIT-IDENTITY:** after 200 steps, `np.array_equal` for each float32
   ex/ey/ez/hx/hy/hz and all 24 psi arrays, new vs baseline. Fixtures:
   uniform CPML-8 open box with a soft source; graded-z CPML-8 box;
   per-face PEC/PMC/CPML box with differing face counts; CPML-4; CPML-16.
   Use 12 interior cells per axis, dx=1 mm, graded z spanning 0.5–1 mm,
   a center soft ez Gaussian pulse. Include a periodic-axis fixture and
   kappa>1 coverage. If any bit moves, **STOP**, report the first differing
   operation and its arithmetic/codegen change; no epsilon-close acceptance
   and no further candidate tuning or GPU timing.
2. **AD parity:** `jax.grad` with respect to dz_profile and eps_r on the
   12-cell graded-z CPML-8 fixture, 200 steps, objective sum of squared
   final E fields. Compare maximum absolute gradient difference divided by
   maximum absolute reference gradient (floor 1e-30), <=1e-6 for each.
   This fixture exercises traced boundary spacing, dt, curl metrics,
   material coefficients and recursive psi; require finite nonzero gradients.
3. No public signature change. Uniform fused fast path and stencil_order=4
   remain untouched. No changes to simulation.py/nonuniform.py step order.
4. Run focused identity/AD gates, then all unit/contracts excluding gpu,
   slow, slow_physics, with the supplied Python/PYTHONPATH and no cache
   provider; run ruff E,F,W ignoring E501,F401,E741,E731,E701,E702,E402.
   Report counts and failures; do not chase the named pre-existing oracles.
5. One RTX 4090 VESSL run after correctness passes: same G1b marginal-cost
   harness (64 -> 1088, three windows, median and max-minus-min spread),
   bare-slow and nu-uniform at 300^3 L=0/4/8/16, plus both at 400^3 L=8.
   Resubmit once only for a failed run. Harvest logs and delete with the
   prescribed harvest script. No push or PR.

## Expected gain and falsifier (before any run)

CPML-8 ideal shell localization predicts
`c_new = c0 + f8*(c8-c0)`: 0.153327 ns bare-slow and 0.166100 ns NU,
about 3.08x and 3.38x speedups. Retaining half the removable full-array
excess gives about 1.51x and 1.54x. Pre-declared expected CPML-8/300 gain:
**1.5–3.4x**. This is a model range, not a measured claim; slab scatter
writes may retain volume traffic. L=4 should benefit at least as much as
L=16 in the ideal model; report all rows without dropping noisy windows.

Timing gate: at 300^3 CPML-8 on **both** lanes, new median minus old
median must exceed **twice the larger of the two measured spreads**.
If either fails, STOP and record that this localized whole-array structure
was not demonstrated to be the cost. Also report this predicate for L=4/16.
L=0 is the no-absorber control: code path unchanged and median difference
must be within twice the larger spread; otherwise comparison is confounded
and no gain is accepted. Expectation held means both CPML-8/300 ratios are
within 1.5–3.4 and timing/control gates pass. Record any fired gate plainly.

Before rows come from `w8b_nu_kernel_ablation_4090.json`. That JSON has
**no bare-slow or nu-uniform 400^3 row**: report those missing before cells
as unavailable, with any older G1 comparison explicitly labeled separately.
Do not substitute graded nu-z or scalar-inv for the requested lane.
