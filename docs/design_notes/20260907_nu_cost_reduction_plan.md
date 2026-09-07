# NU cost-reduction lane (G) — plan and pre-measurements

**Status:** plan (2026-09-07). Code work starts AFTER the accuracy/autodiff lane
(branch `feat/nu-band-accuracy-ad`, note
`docs/design_notes/20260907_nu_band_accuracy_ad_predeclaration.md`) has
reported, because G2 and G3 take their accuracy comparator from it (PI rule:
one lane at a time; a policy that trades cells for accuracy is decided on a
measured accuracy curve, not on cost alone).
**Branch:** `feat/nu-cost-reduction` (from `feat/nu-band-profile` @ 7c6e3997).

## What the user observed and what it is

"The NU path gets very slow." Measured on this tree (CPU, 65^3, CPML 8):

| lane | ms/step | compile |
|---|---|---|
| uniform runner | 4.64–5.28 | 1.5 s |
| NU runner, uniform-valued `dz_profile` (same mesh, same dt) | 4.69–4.95 | 0.3 s |

Per-step ratio 0.94–1.0x. The GPU table in `docs/agent/gpu-throughput.mdx`
(2026-08-25, 300^3 CPML, same harness) shows the NU kernel at −8.5 % (H200),
−16 % (RTX 4090), −43 % (A6000) against the uniform kernel. So the kernel is
not where "very slow" comes from.

Where it comes from is `dt` — the NU `dt` is the global minimum cell's CFL
(`make_nonuniform_grid`; confirmed on a 2:1 graded run: `Result.dt`
1.3482e-12 s = the 0.5 mm cell's value, not the 1 mm one) — times the cell
count. On the 5-layer PCB stack (core 0.8 / prepreg 0.1 / … mm, dx 0.2 mm):

| mesh | nz | dt | steps | cells | steps x cells |
|---|---|---|---|---|---|
| uniform 200 um (does not resolve the prepreg) | 20 | 381 fs | x1 | x1 | **x1** |
| auto-z `_make_dz_profile` (thirds rule, dz_min 8.33 um) | 115 | 27.5 fs | x13.9 | x5.75 | **x80** |
| `make_band_profile` cap 1.4, min_cells 4 (dz_min 25 um) | 89 | 81.3 fs | x4.7 | x4.45 | **x21** |

The thirds rule's 1/3 sub-cell alone is x3.8 of the x80. The remaining x21
is the cost of resolving a 100 um layer with 4 cells at all — physics, not
overhead — and it is the lever G3 prices.

A third contributor is common to both lanes and not NU's: production runs
with per-step port DFTs and probes run ~7x below the bare kernel on the H200
(`gpu-throughput.mdx`, "Kernel rate is not production rate").

## Items, in order

### G1 — NU kernel parity with the uniform fast path (pure refactor, bit-identity gated)

`update_e_nu` (`rfx/core/yee.py:456`) recomputes `sigma*dt/(2 eps)`, `ca`,
`cb` from the material arrays at EVERY step and the NU scan applies
`apply_pec` / `apply_pec_mask` as separate passes; the uniform lane's
`update_he_fast` uses `precompute_coeffs` (coefficients once, PEC baked in,
H+E fused). Plan: a `precompute_coeffs_nu` (ca, cb without the spacing, which
stays in the per-axis `inv_*` arrays) and an `update_he_nu_fast`, selected by
the same `use_fast_he` predicate the uniform lane uses (no dispersive / aniso
/ Kerr / sheet paths — those keep the current stepper).

Gate: field arrays after N = 200 steps on the W6/PCB fixture and on a
CPML+PEC-mask fixture must be **bit-identical** (`np.array_equal` on ex..hz,
float32) to the current stepper — the coefficient arithmetic is the same ops
in the same order, only hoisted; if any bit moves, STOP and show the op that
reordered. AD: `jax.grad` w.r.t. `dz_profile` and w.r.t. eps on the fast
path equals the slow path to 1e-6 relative (the coefficients are still traced
functions of the same inputs).

Measurement: `scripts/diagnostics/gpu_throughput_bench.py` on the RTX 4090
(`scripts/vessl_nu_cost_gpu_bench.yaml`), BEFORE (this tree, staged commit
5992675a, VESSL run 369367259061) and AFTER, same card, same harness; report
the nu-z / bare ratio at 100^3..400^3. Pre-declared expectation, not a gate:
the nu-z rate moves toward bare; the win is bounded by the 16 % gap the
08-25 table shows on this card, so anything claimed above that is an
instrument error.

### G2 — take the thirds rule out of the auto-z path (policy, accuracy-gated)

`_make_dz_profile` applies `apply_thirds_rule` at every dielectric feature
boundary (dielectric|air and dielectric|dielectric) and never at a conductor —
`analyze_features` drops PEC bodies from `z_features` — although the rule's
own docstring reasons about a conductor side (openEMS "thirds" for metal
edges). On a dielectric interface that already sits on a node plane the split
buys nothing the ratio law needs and costs the 1/3 sub-cell that sets `dt`
for the whole column (x3.8 on the PCB stack; x9.8 in nz on a thin-between-
thick stack, see the band-profile note's #931 paragraph).

Gate (from the accuracy lane, frozen there): arm AZ (auto-z with thirds) vs
arm MB (`make_band_profile`, no thirds, interfaces on nodes) on the stratified
cavity ladder. Decision rule, declared now: remove the thirds step from
`_make_dz_profile` if |err_MB| <= |err_AZ| x 1.1 at matched finest non-thirds
cell across the ladder (i.e. the split does not buy accuracy at a dielectric
interface); keep it and document the cost if AZ is better by more than that.
`apply_thirds_rule` stays public and unchanged either way; metal edges get
their node plane from the #931 ownership contract, not from a split cell.
Locked values that move (`test_make_dz_profile_applies_thirds_rule`, the
#763 demo block `[63.5, 63.5, 63.5, 42.333, 21.167]` um) are re-pinned with
this note's rule as provenance, never silently.

### G3 — thin-layer cell policy from the measured error law (policy, accuracy-gated)

`min_cells_per_feature = 4` is a constant. The accuracy lane's A1 ladder
measures the resonance error of a thin dielectric layer resolved with 2 / 4 /
8 cells (scales 0.5 / 1 / 2). Plan: fit the error-vs-cells law from that
table, expose `auto_configure(..., thin_layer_cells=)` with the `accuracy`
presets ('draft' / 'standard' / 'high') mapping to the cell count that meets
each preset's declared error, and state the dt consequence in the mesh
report (`SimConfig.summary`) so the user sees the steps x cells price before
running. No default changes until the law is measured; the preset numbers are
derived from A1's table and cited.

#### G1 baseline — measured (RTX 4090, VESSL run 369367259061, staged 5992675a, 2026-09-07)

Log: `~/Documents/vessl-run-logs/369367259061_rfx-nu-cost-gpu-bench-baseline.log`
(run deleted after harvest). Marginal-cost differencing, steps 64 -> 1088,
median of 3 windows:

| n^3 | bare Mcells/s | nu-z Mcells/s | nu-z / bare |
|---|---|---|---|
| 100 | 2509 (spread 1165) | 2465 (spread 1517) | 0.98 (noise-dominated) |
| 200 | 1761 (spread 934) | 1667 (spread 80) | 0.95 |
| 300 | 2210 (spread 225) | 1788 (spread 83) | **0.81** |
| 400 | 1736 (spread 78) | 1569 (spread 59) | **0.90** |
| 512 | OOM | OOM | — |

Consistent with the 2026-08-25 table on this card (2119 / 1784 at 300^3).
The G1 comparator is the 300^3 and 400^3 rows (spreads < 5 %); the 100^3
and 200^3 bare rows have spreads of 40-50 % and are not usable as a gate.
The JSON was lost to a PermissionError (checkout not writable by the
container); the yaml now runs a copy of the bench from the writable run
directory.

### G1 ablation — pre-declared (2026-09-07, written BEFORE the first GPU run)

Premise check (lead, 2026-09-07): under the CPML-8 bench condition the
uniform lane's fused fast path is gated OFF (`_fast_eligible` requires no
CPML, `rfx/simulation.py` ~2000-2025), so BOTH lanes run their slow
steppers (`update_h`/`update_e` vs `update_h_nu`/`update_e_nu`) and both
recompute `sigma_dt_2eps`/`ca`/`cb` every step. Whether XLA hoists that
loop-invariant arithmetic out of the `lax.scan` body is UNKNOWN. The
0.81 / 0.90 gap therefore has to be ATTRIBUTED, not assumed, before any
refactor: the instrument is
`validation/research/nu_cost/w8_nu_kernel_ablation.py`
(`scripts/vessl_nu_cost_ablation.yaml`, RTX 4090, same harness as the
baseline: marginal-cost differencing 64 -> 1088, 3 windows, median +
spread, fp32, one soft source, `skip_preflight=True` which is host-side
fixed cost the differencing removes). No `rfx/` source changes; every arm
is an in-process monkeypatch on the module attribute the step body binds.

Arms and fixtures:

| arm | lane / mesh | patch | reference | bit-identity expected | fixtures |
|---|---|---|---|---|---|
| bare | uniform / uniform | — | — | — | cpml8 300, 400; pec 300 |
| nu-uniform | NU / uniform-valued profile | — | — | — | cpml8 300, 400; pec 300 |
| nu-z | NU / 4:1 graded | — | — | — | cpml8 300, 400; pec 300 |
| nu-z-hoist | NU / graded | `update_e_nu` coefficients computed once (`ensure_compile_time_eval`, same expression, same op order) | nu-z | yes | cpml8 300, 400 |
| bare-hoist | uniform / uniform | same hoist on `update_e` | bare | yes | cpml8 300, 400 |
| nu-z-nopec | NU / graded | `apply_pec` skipped | nu-z | **no** (physics differs; attribution only) | pec 300 |
| nu-z-scalar-inv | NU / uniform-valued | six `inv_*` broadcast vectors -> scalars of the same float32 value | nu-uniform | yes | cpml8 300, 400 |

Bit-identity check, mandatory and in-process BEFORE any timing: each
patched arm runs 64 steps on a 96^3 box of its fixture with and without
its patch; `np.array_equal` on ex..hz (float32). An arm whose patch moves a
bit is timed but flagged and can never become an implementation candidate.
The instrument also records whether the hoist actually produced concrete
arrays (`hoisted`) and whether the uniform fused fast path was traced
(`fast_path_seen`; on GPU it will be for `bare` on the pec fixture — that
arm is then the fused target, not the slow path, and is read as such).

Attribution table (filled from the JSON, cost units c = 1/rate normalised
by c_bare, one row per fixture/n):

| fixture / n | nu-z / bare | gap = c_nuz/c_bare − 1 | graded access (nu-z − nu-uniform) | NU code path (nu-uniform − bare) | broadcast (nu-uniform − scalar-inv) | hoist total (nu-z − nu-z-hoist) | hoist common (bare − bare-hoist) | hoist NU-specific | PEC pass (nu-z − nopec, pec fixture) | sum check |
|---|---|---|---|---|---|---|---|---|---|---|
| cpml8 / 300 | | | | | | | | | — | |
| cpml8 / 400 | | | | | | | | | — | |
| pec / 300 | | | | | — | — | — | — | | |

Implementation rule (declared now): an arm becomes an implementation
candidate only if it is bit-identical AND beats its reference by more than
2x the larger of the two spreads AND by >= 3 %, on BOTH 300^3 and 400^3
(cpml8). Anything less is noise or not worth a refactor under the
bit-identity gate. `nu-z-nopec` is excluded by construction (not physics-
identical); it only prices the separate pass.

Sum check (declared now): the disjoint pieces graded access + broadcast +
hoist NU-specific (+ PEC pass on the pec fixture), each clipped at 0, must
sum to no more than the measured gap (1/0.81 − 1 = 0.23 at 300^3,
1/0.90 − 1 = 0.11 at 400^3) plus the spread tolerance (sum of the
contributing rows' spreads in cost units). If they sum to more, the
instrument is wrong and nothing from it is used. The G1 ceiling stands:
no arm can be credited with more than the gap itself.

Expectations, not gates: the NU code path piece and the graded-access
piece together equal the gap exactly (telescoping); the hoist is either a
common win (bare-hoist moves too) or nothing (XLA already hoists) — the
instrument decides which; the broadcast piece is expected small (the 1-D
reads fuse into the elementwise kernel).

Smoke (CPU, `--smoke`, 32^3, 64-step identity check, run before commit):
nu-z-hoist / bare-hoist / nu-z-scalar-inv bit-identical to their
references (0 differing elements, `hoisted=True`), nu-z-nopec not identical
(expected). GPU numbers: none yet — this section is closed to edits once
the run starts; results go in a new "G1 ablation — measured" section.

### G1 ablation — measured (2026-09-07, VESSL run 369367259106, RTX 4090)

Run: staged commit 5c1b08c7, `scripts/vessl_nu_cost_ablation.yaml`, jax
0.4.33 (nvcr jax:24.10), cuda:0, 64 -> 1088 steps, 3 windows, fp32,
`skip_preflight=True`. Wall time 16:55 -> 17:31 (36 min). Harvested to
`~/Documents/vessl-run-logs/369367259106_rfx-nu-cost-g1-ablation.log`, run
deleted. JSON: `validation/research/nu_cost/results/w8_nu_kernel_ablation_4090.json`.
No OOM, no SKIPPED row; all 16 arm/size rows produced a number.

Bit-identity (96^3 x 64 steps, np.array_equal on ex..hz, before any timing):

| arm | reference | identical | expected | result |
|---|---|---|---|---|
| nu-z-hoist | nu-z | True | True | OK, max diff 0, hoisted=[True] |
| bare-hoist | bare | True | True | OK, max diff 0, hoisted=[True] |
| nu-z-scalar-inv | nu-uniform | True | True | OK, max diff 0, six inv_* = 999.9999389648438 |
| nu-z-nopec | nu-z | False | False | OK (attribution only), max diff 1.489e-16 |

Rates (Mcells/s, median of 3 windows, spread in Mcells/s). `hoisted` was
True on all 9 traces of each hoist arm; `fast_path` was False on every
cpml8 row and True only for bare on the pec fixture (as pre-declared).

| fixture / n | bare | nu-uniform | nu-z | nu-z-hoist | bare-hoist | nu-z-scalar-inv | nu-z-nopec |
|---|---|---|---|---|---|---|---|
| cpml8 / 300 | 2121.6 (3.3) | 1780.9 (5.2) | 1779.2 (12.6) | 1779.7 (17.1) | 2147.6 (20.7) | 2123.3 (24.2) | — |
| cpml8 / 400 | 1767.4 (8.7) | 1582.2 (2.1) | 1584.7 (1.1) | 1797.4 (12.5) | 1728.4 (1.0) | 1764.6 (14.6) | — |
| pec / 300 | 8664.0 (36.4) fused | 10039.5 (29.5) | 10018.0 (82.7) | — | — | — | 10143.8 (48.6) |

nu-z / bare: 0.839 at 300^3 (baseline run 369367259061: 0.81; bare here is
2121.6 vs 2210 there, nu-z 1779 vs 1788), 0.897 at 400^3 (baseline 0.90).

Attribution (cost units c = 1/rate, normalised by c_bare; values from the
JSON `attribution` block, rounded to 3 decimals):

| fixture / n | nu-z / bare | gap | graded access (nu-z − nu-uniform) | NU code path (nu-uniform − bare) | broadcast (nu-uniform − scalar-inv) | hoist total (nu-z − nu-z-hoist) | hoist common (bare − bare-hoist) | hoist NU-specific | PEC pass (nu-z − nopec) | sum of clipped pieces | gap + spread tol. | sum check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cpml8 / 300 | 0.839 | 0.193 | 0.001 | 0.191 | 0.192 | 0.000 | 0.012 | −0.012 | — | 0.193 | 0.193 + 0.058 = 0.250 | OK |
| cpml8 / 400 | 0.897 | 0.115 | −0.002 | 0.117 | 0.115 | 0.132 | −0.023 | 0.155 | — | 0.270 | 0.115 + 0.025 = 0.140 | **FAIL** |
| pec / 300 | 1.156 | −0.135 | 0.002 | −0.137 | — | — | — | — | 0.011 | 0.013 | −0.135 + 0.021 = −0.114 | FAIL (gap negative: bare row is the fused path, not a slow-vs-slow gap; the sum check is ill-posed on this fixture as declared) |

Candidate rule applied mechanically (bit-identical AND gain >= 3 % AND
delta > 2x larger spread, on BOTH cpml8/300 and cpml8/400):

| arm | identical | cpml8/300 | cpml8/400 | candidate |
|---|---|---|---|---|
| nu-z-hoist | yes | +0.03 % (0.6 vs 2x17.1) fail | +13.4 % (212.8 vs 2x12.5) pass | no (one size only) |
| bare-hoist | yes | +1.2 % (26.0 vs 2x20.7) fail | −2.2 % (−39.0) fail | no |
| nu-z-nopec | no | — | — | excluded by construction (pec/300: +1.3 %, 125.8 vs 2x82.7, fail anyway) |
| nu-z-scalar-inv | yes | +19.2 % (342.4 vs 2x24.2) pass | +11.5 % (182.4 vs 2x14.6) pass | passes the candidate rule |

What the numbers say, without adjectives:

* Graded access costs nothing: nu-z = nu-uniform at both sizes (0.001 and
  −0.002 cost units, inside spread). The whole gap is the NU code path on
  a uniform-valued mesh (0.191 / 0.117 = the gap at both sizes, telescoping
  as expected).
* The six 1-D `inv_*` broadcast multiplies in the two NU curl kernels
  (the scalar-inv patch touches only `update_h_nu`/`update_e_nu`, not the
  NU CPML variant) account for the gap at both sizes: 0.192 of 0.193 at
  300^3, 0.115 of 0.115 at 400^3. With scalars the NU lane runs at the bare
  rate (2123.3 vs 2121.6; 1764.6 vs 1767.4).
* The coefficient hoist is size-inconsistent: nothing at 300^3 on either
  lane (nu-z-hoist = nu-z within 0.6 Mcells/s; bare-hoist +26, inside 2x
  spread), but at 400^3 nu-z-hoist gains 212.8 (13.4 %, above bare) while
  bare-hoist LOSES 39.0 (−2.2 %, delta > 2x spread). Both hoist arms embed
  two full-size ca/cb constants (~256 MB each at 400^3); the 400^3 behaviour
  is a size-dependent codegen effect of that embedding, not a per-step
  arithmetic saving (which would have shown at 300^3 too). Cause not
  determined by this run.
* The separate `apply_pec` pass prices at 0.011 cost units on the pec
  fixture, 125.8 Mcells/s vs spread 82.7: not attributable above spread.
* Observation outside G1's target, recorded as a number: the pec fixture
  (no CPML) runs at ~10,000 Mcells/s on the NU slow path vs ~1,780 with
  CPML 8 at the same 300^3, i.e. the CPML-8 bench step costs ~5.6x the
  no-CPML step. The G1 gap is measured inside a CPML-dominated step. The
  bare row on pec took the fused fast path (8664.0, fast_path=True) and was
  SLOWER than the NU slow path (10039.5); that row is the fused target and
  is read only as such.

Sum check consequence (rule as written above: "If they sum to more, the
instrument is wrong and nothing from it is used"): the check FAILS at
400^3. The failure is not spread: the hoist NU-specific piece (0.155) and
the broadcast piece (0.115) are not disjoint at 400^3 — each alone
recovers the whole gap — while the hoist piece is 0 at 300^3. The
"disjoint pieces" model the sum check assumes is therefore wrong at 400^3,
and the rule gives no way to tell an instrument error from a
size-dependent XLA effect. A combined hoist+scalar-inv arm would decide
that; it was not pre-declared and is not added post hoc here.

**Decision (pre-declared rules applied mechanically): STOP — sum check
failed at 400^3 (0.270 > 0.140); nothing from this run is used for an
implementation decision.** For the record: nu-z-scalar-inv is the only
arm that met the candidate rule (bit-identical, +19.2 % / +11.5 %, both
> 2x spread), and it is blocked by the sum check alone, not by spread.
Next instrument revision (lead decision, not taken here): add the
combined arm and a smaller-constant hoist variant (e.g. hoist only
`sigma_dt_2eps`, or hoist as scan carry/closure instead of an embedded
constant) so the 400^3 hoist effect is either explained or disappears,
re-declare the sum check with the pieces it actually tests, then rerun.
The run did not fail, so the one-resubmission clause was not used.

### G1b — pre-declared (2026-09-07, written BEFORE the G1b GPU run)

Why a second instrument: the first run's sum check failed at 400^3 because
its "disjoint pieces" model was wrong there (the hoist piece and the
broadcast piece each recovered the whole gap; see "G1 ablation — measured").
G1b drops the sum check and judges every arm ONLY against its own
reference. Instrument: `validation/research/nu_cost/w8_nu_kernel_ablation.py
--suite g1b` (`scripts/vessl_nu_cost_ablation_g1b.yaml`, RTX 4090, same
harness: marginal-cost differencing 64 -> 1088, 3 windows, median + spread,
fp32, one soft source, `skip_preflight=True`). No `rfx/` source changes;
every arm is an in-process monkeypatch on the module attribute the step
body binds. The G1 suite stays runnable unchanged (`--suite g1`, default).

Part A — cpml8 at 300^3 and 400^3. Timing reference for every arm = **nu-z
of the same run** (same fixture, same n). Identity reference = the unpatched
arm on the same mesh.

| arm | mesh | patch | identity ref | bit-identical expected | extra full-size fp32 constants |
|---|---|---|---|---|---|
| nu-z | 4:1 graded | — | — | — | 0 |
| nu-z-scalar-inv | uniform-valued | six `inv_*` vectors -> scalars (repeat of G1; expected to reproduce +19.2 % / +11.5 %) | nu-uniform | yes | 0 |
| nu-z-combo | uniform-valued | scalar-inv in both curls + the `update_e_nu` coefficient hoist (`ensure_compile_time_eval`, same expression as nu-z-hoist) | nu-uniform | yes (each half was) | 2 (ca, cb) |
| nu-z-foldinv | 4:1 graded | `inv_*` FOLDED into the coefficients once: E uses `cb*inv_dx[:,None,None]`, `cb*inv_dy[None,:,None]`, `cb*inv_dz[None,None,:]`; H uses `(dt/mu)*inv_dx_h`, `..._h` likewise; the curls multiply the differences by those full 3-D arrays: `ex = ca*ex + (cby*dHz - cbz*dHy)` instead of `ca*ex + cb*(dHz*inv_dy - dHy*inv_dz)` | nu-z | **no** (products re-associated) -> EPSILON gate | 7: the per-component 6 E + 6 H arrays collapse to 3 + 3 DISTINCT arrays because ca/cb are isotropic (`cb_ex_y` and `cb_ez_y` are both `cb*inv_dy`), plus the hoisted `ca` |
| nu-z-inv3d | 4:1 graded | six `inv_*` vectors pre-broadcast to MATERIALIZED full 3-D arrays once (`jnp.array(jnp.broadcast_to(...))` under `ensure_compile_time_eval`); same multiplies, same op order | nu-z | yes | 6 |

Constant memory the arms embed (grid is (n+1+2L)^3: 317^3 at 300 / 417^3
at 400, one fp32 array = 127.4 / 290.0 MB; the unpatched scan already
closes over 3 material arrays = 0.38 / 0.87 GB): combo +255 / +580 MB;
inv3d +0.76 / +1.74 GB (total 1.15 / 2.61 GB); foldinv +0.89 / +2.03 GB
(total 1.27 / 2.90 GB). The first run compiled 1.45 GB of constants at
400^3 (nu-z-hoist). A 400^3 row that fails to compile or OOMs is recorded
`SKIPPED` with the exception text; that is a result about the design
(a real implementation would close over the same arrays), and the arm
then fails the both-sizes rule by construction. The JSON records
`extra_const_bytes` and `jax.devices()[0].memory_stats()` (bytes_in_use,
peak) per row.

Epsilon gate for nu-z-foldinv (declared number: **4 ulp**): after the
identity run (96^3, 64 steps, float32), for every component,
`max|new − old| <= 4 * ulp(max|family field|)`, where the family maximum
is max|E| over ex, ey, ez for an E component and max|H| over hx, hy, hz
for an H component (`np.spacing(np.float32(max))`). The family
normalisation is declared because hz is zero by symmetry in this fixture
(ez point source, symmetric cube): on the CPU smoke max|hz| = 1.4e-3
against max|hx| = 2.6e4, so its own maximum is rounding residue and a gate
against it is ill-posed (3.1e7 "ulp"). The own-maximum ulp is recorded
alongside. CPU smoke, 32^3 x 64 steps: worst 3.62 ulp (hy), E components
<= 1 ulp — inside the gate with a thin margin; the GPU (different FMA
contraction) decides, and a value above 4 makes foldinv a non-candidate
with the number reported.

Rule for part A (each arm against nu-z, no sum check): an arm PASSES if
its delta > 2x the larger of the two spreads AND its gain >= 3 % at BOTH
300^3 and 400^3. `nu-z-foldinv` is the only implementable design (it runs
on the graded mesh); it becomes the implementation candidate only if it
passes AND meets the epsilon gate. scalar-inv, combo and inv3d run on the
uniform-valued mesh or materialize arrays nobody would ship: they are
attribution arms, judged by the same delta rule, never candidates.

Pre-declared readings (from `evaluation.readings`, both sizes):

* combo vs scalar-inv (same mesh): |delta| <= 2x spread -> hoist and
  broadcast are the SAME cost (not additive; the 400^3 hoist effect of the
  first run is then the broadcast by another route); combo faster by
  > 2x spread -> additive; combo slower -> the hoist costs on top.
* inv3d vs nu-z: |delta| <= 2x spread -> removing the broadcast op buys
  nothing, the cost is the coefficient read; inv3d faster -> the broadcast
  op costs more than a full-array read; inv3d slower -> full-array reads
  cost more than the broadcast, the broadcast op itself is cheap and the
  scalar-inv gain is a fusion effect, not a traffic effect.

Part B — CPML-layer cost ladder at 300^3, layers L = 0 / 4 / 8 / 16, both
lanes. Uniform lane arm `bare-slow`: the fused fast path is FORCED OFF by
swapping the name `jax` inside `rfx.simulation` for a proxy object whose
`default_backend()` returns `"cpu"` and whose every other attribute
delegates to the real module. That name is read exactly once for this
purpose: `rfx/simulation.py:1998` `_on_gpu = jax.default_backend() !=
"cpu"`, and `_on_gpu` feeds only `use_fast_he = _fast_eligible and
_on_gpu and stencil_order == 2` (line 2025); no physics flag
(`use_cpml`, `use_tfsf`, ...) is touched, so the layers-0 row runs the
same `update_h`/`update_e` stepper as the layers-4/8/16 rows. The probe
on `update_he_fast` must report `fast_path_seen=False` on every
`bare-slow` row. NU lane arm: `nu-uniform` (uniform-valued profile).
Context row: `bare` (fused) at L = 0 — its rate against `bare-slow` at
L = 0 is the forcing effect, recorded, not judged; the two paths differ in
coefficient op order by construction (`jnp.float32(dt/(MU_0*dx))/mu_r` vs
`(dt/mu)*curl`), so `bare-slow` has no identity gate, only a recorded
identity check (trivially identical on CPU, where the fast path is never
taken).

Expectation derived before the run. The absorber PADS the grid: the
cpml8/300 rows of the first run have 31,855,013 = 317^3 cells and pec/300
has 301^3 (JSON `cells`), so the absorbing fraction of the allocated grid
is f(L) = 1 − ((n+1)/(n+1+2L))^3 = **0.076 / 0.144 / 0.261** at L =
4 / 8 / 16 (the inside-the-box formula 1 − (1 − 2L/300)^3 gives
0.078 / 0.152 / 0.287 — same picture, the padded values are the ones the
rate metric sees). A CPML pass whose work is proportional to the cells it
covers, at k times a plain cell's cost, gives step cost c(L) = c(0)·(1 +
k·f(L)); with k = 2 that is 1.15x / 1.29x / 1.52x — i.e. **<= 1.3x at 8
layers**. The first run measured ~5.6x (pec/300 10,039 vs cpml8/300 1,781
Mcells/s on the NU lane). Reading of `apply_cpml_e` (`rfx/boundaries/
cpml.py`): per step it forms `dt/(materials.eps_r*EPS_0)` over the FULL
array, takes `_shift_bwd(state.hz, 0)` over the FULL array before slicing
`[:n_x]`, and applies four `.at[slab].add()` per component per axis = 24
dynamic-update-slices on E plus 24 on H per step, each a full-array
operand — whole-array work independent of L if XLA does not fuse them in
place. That is the finding the ladder looks for.

Model reported (declared): cost per cell-step c(L) = 1/rate in ns, fitted
by least squares as **c = a + b·f(L)** on the L = 4 / 8 / 16 rows (the CPML
op present), and also c = a' + b'·L; `a` is the extrapolated cost with the
op present at zero absorbing fraction and c(0) the measured cost with the
op absent. Derived numbers per lane: `intercept_excess` = (a − c(0))/c(0)
(the L-independent part of the CPML op in plain-step units), `rho_16/4`
= (c(16) − c(0))/(c(4) − c(0)) (proportional expectation f(16)/f(4) =
3.46; whole-array work gives ~1.0), `rho_8/4` (expectation 1.90), and
c(L)/c(0). Declared reading: **whole-array work in the CPML op** if
`intercept_excess` >= 1.0 OR `rho_16/4` <= 1.5; otherwise the cost scales
with the absorbing fraction. No candidate comes out of part B; it prices
where the step goes and whether a CPML refactor, not an NU one, is the
lever.

Smoke (CPU, `--smoke --suite g1b`, 32^3, 64-step checks, run before
commit): nu-z-scalar-inv, nu-z-combo (`hoisted=True`), nu-z-inv3d
(`materialized=True` x6) bit-identical to their references (0 differing
elements); nu-z-foldinv not identical as expected, epsilon gate PASS at
3.62 ulp worst (`folded_e`/`folded_h`/`hoisted` all True); bare-slow
`slow_path_forced=True`, `fast_path_seen=False`; all 14 timing rows and
both ladder fits produced. The G1 suite smoke still runs every original
arm. GPU numbers: none yet — this section is closed to edits once the run
starts; results go in a new "G1b — measured" section.

## Not pursued

Local time stepping / domain-wise dt — excluded by the support matrix (late-
time interface instability, Xiao et al. TAP 55(7):1981, 2007). Not revisited
here.

## Reproduce the pre-measurements

```
# per-step parity (CPU): 65^3, CPML 8, uniform vs NU with a uniform-valued profile
PYTHONPATH=<tree> python - <<'EOF'
import numpy as np, time
from rfx import Simulation, GaussianPulse
N=48; dx=1e-3; L=N*dx
for nu in (False, True):
    kw = dict(freq_max=10e9, domain=(L,L,L), dx=dx, boundary="cpml", cpml_layers=8)
    if nu: kw["dz_profile"] = np.full(N, dx)
    s = Simulation(**kw)
    s.add_source((L/2,L/2,L/2), "ez", waveform=GaussianPulse(f0=5e9, bandwidth=0.5), amplitude_kind="current")
    s.add_probe((L/2+5*dx, L/2, L/2), "ez")
    s.run(n_steps=50); t=time.perf_counter(); s.run(n_steps=400); print(nu, (time.perf_counter()-t)/400*1e3, "ms/step")
EOF
# dt / cell cost table: docs/design_notes/20260907_nu_band_profile_predeclaration.md fixture,
# dt = 0.99 / (c0 * sqrt(1/dx^2 + 1/dy^2 + 1/dz_min^2))
```
