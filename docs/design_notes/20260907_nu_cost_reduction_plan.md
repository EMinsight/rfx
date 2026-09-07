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
