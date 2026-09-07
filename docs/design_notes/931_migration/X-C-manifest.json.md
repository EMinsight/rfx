# X-C → `validation/crossval/manifest.json` (cases 14, 15, 16, 17)

Owner of this text: the X-C crossval migration agent (issue #931, branch
`feat/931-crossval-C`). `manifest.json` itself is owned by the merge/ingest
agents, so the replacement text lives here.

Two of the four edits are ready to apply now. The cv15 `claim_scope` carries
NUMBERS that come from a re-solve; the rules for filling them are written out
below, and nothing here may be hand-translated from the old figures.

---

## 1. `id: 14_rect_cavity_pozar` — `claim_scope`

One clause changes. Find:

> Gated modes (re-declared 2026-08-31, issue #812): Gate 0 wall registration
> `|(n-1)*dx - (a,b,d)| <= 1e-9 m` on every axis;

replace with:

> Gated modes (re-declared 2026-08-31, issue #812; Gate 0 re-based 2026-09-07,
> issue #931): Gate 0 wall registration `|realized wall separation - (a,b,d)|
> <= 1e-9 m` on every axis, where the separation is MEASURED — the shipped
> `apply_pec` / `apply_pec_faces` are applied to an all-ones state and the
> planes are read back through `rfx.boundaries.pec.realized_wall_planes`, which
> must be exactly `{0, n-1}` per axis, and the same measured planes feed the
> Yee oracle. Before #931 this gate computed `(n-1)*dx` from the grid SHAPE and
> would have kept passing under any change to how a PEC wall is realized. The
> 1e-9 m tolerance is unchanged, so this is a strict tightening;

Rationale for the reviewer: domain-boundary PEC is fenced out of the ownership
contract (design note §1.8) and keeps its own convention. cv14 is the case that
proves the fence holds, and until #931 nothing measured it.

## 2. `id: 16_pec_sphere_mie_ka_sweep` — `claim_scope`

Append one sentence to the end of the existing `claim_scope` (nothing else in
it moves — no cv16 number changes under #931):

> CONDUCTOR MODEL (issue #931): this case's sphere is a high-sigma MATERIAL
> FILL (`rasterize(Sphere, eps_r 1.0, sigma 1e7)` straight onto
> `MaterialArrays`), never a `Simulation.add(..., material='pec')` entry, and
> the lattice ownership contract FENCES that model out (§1.8): a sigma fill is
> a lossy volume model on NODE samples and is unchanged here, while a declared
> PEC volume is sampled at cell CENTRES (§1.1). The two are checked against each
> other at build time instead of being assumed equal —
> `assert_conductor_model()` refuses to quote an RCS unless the fill still
> equals `Sphere.mask(grid)`, and reports the centre-sampled cell set beside
> it. Measured at this case's own gated points: ka = 0.50 coarse N 1082 → 1123
> (a_eff/a 0.988032 → 1.000357, 259 cells differing), fine ka = 2.00 N 9264 →
> 9339 (0.998319 → 1.001006, 957 cells differing). Bringing this case under the
> contract would therefore move a_eff by ~1.2 % at ka = 0.5, move the Mie
> reference leg with it, and require the fixture and both gate constants to be
> regenerated from a re-solve; that is not done here.

## 3. `id: 17_dielectric_sphere_mie` — `claim_scope`

Append one sentence (no cv17 number changes; this case is the #931 control):

> DIELECTRIC CONTROL (issue #931): the lattice ownership contract changes PEC
> VOLUME sampling from node to cell-centre and leaves dielectric sampling
> untouched (§1.1), so every number in this case must be bit-identical across
> that change. `tests/crossval/test_rcs_dielectric_sphere_mie_gates.py` asserts
> it at every gated bin: exactly two distinct eps values with the
> non-background one at the declared 2.56, and an occupied-cell count equal to
> the NODE-sampled shape mask (1082 at ka = 0.5, the same sphere on the same
> mesh as case 16). If this count ever picks up centre sampling, every number
> in this case is a different sphere.

## 4. `id: 15_patch_antenna_rt5880` — `claim_scope` (numbers pending the re-solve)

### 4a. The mechanism clause — apply as written

Find, inside `claim_scope`:

> Post-#740 (ground wall realized at the substrate floor via two_plane; the
> pre-fix one-plane ground left a vacuum cell in the cavity and read +6.09% vs
> openEMS) rfx reads 0.69% LOW vs openEMS (2.3139 vs 2.330 GHz) and -4.21% vs
> the analytic anchor, openEMS -3.54%: both solvers sit on the same side of the
> closed form by a similar margin; that direction is discretisation, reported
> not hidden.

replace with (`<...>` are filled per §4b — do not translate the old digits):

> Post-#931 (lattice ownership contract) BOTH conductors are declared SHEETS —
> zero-thickness PEC `Box`es on the two substrate faces — which is the same
> structure both openEMS legs build (`AddBox` with start z == stop z at 0 and
> at h). The feed spans the full substrate, standing ON the ground sheet plane,
> matching the openEMS `AddLumpedPort` span; a port on a conductor's node plane
> is galvanic, not "inside PEC" (#929). This replaces three repairs of one
> undeclared sheet rule: the `two_plane=True` ground drawn a cell below the
> floor (#740, itself the remedy for the #693 vacuum ground cell that read
> +55.0 % electrical thickness and +6.09 % vs openEMS, preserved as
> `validation/crossval/_15_patch_results/rfx_one_plane_ground_b29f9de7.json`),
> the one-cell patch `Box` whose far wall at 11.9062 mm was suppressed only by
> a realization default, and a feed held one cell short of the patch so
> coupling was capacitive only (#556). `assert_realized_stack()` now reads the
> contract's one edge set (`realized_pec_edge_masks` /`realized_wall_planes`)
> and refuses to quote f0 unless the realized wall planes over the patch
> footprint are EXACTLY {z_sub_lo, z_sub_hi} — no wall at k_patch+1 — and the
> assembled eps_r still holds exactly the two declared values, a sheet owning
> no cell and writing no material. The #768 leg is preserved verbatim as
> `validation/crossval/_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json`
> as the before side. Measured on the regenerated leg: rfx reads <D_OE> vs
> openEMS (<F_RFX> vs 2.330 GHz) and <D_AN> vs the analytic anchor, openEMS
> -3.54%; <SAME_SIDE_CLAUSE>. That direction is discretisation, reported not
> hidden.

### 4b. Filling `<D_OE>`, `<F_RFX>`, `<D_AN>`, `<SAME_SIDE_CLAUSE>`

Read them from the REGENERATED `_15_patch_results/rfx.json` and
`_15_patch_results/openems.json` (unchanged), computed the way `compare()`
computes them — do not carry any digit over from the old text:

* `<F_RFX>` = `rfx.json::f_primary_hz` in GHz to 4 significant figures;
* `<D_OE>` = `|f_rfx - f_oe| / f_oe` as a percentage to 2 dp, with the word
  `LOW` or `HIGH` per the sign (`f_rfx < f_oe` → LOW);
* `<D_AN>` = `(f_rfx - f_analytic) / f_analytic` as a signed percentage to 2 dp
  (`rfx.json::f_analytic_hz`);
* `<SAME_SIDE_CLAUSE>` = "both solvers sit on the same side of the closed form
  by a similar margin" ONLY if `<D_AN>` and openEMS's -3.54 % have the same
  sign and differ by less than 2 percentage points. If they do not, say what
  was measured instead; do not keep the sentence.

### 4c. The #812 STOP paragraph — keep, with one added sentence

The whole "ISSUE #812 (2026-09-01, round 2)" paragraph stays as written: the
f0 gate's blindness to the #740 realization, the mode-pair ratio band that was
tried and rejected, and the honest STOP. Its evidence (the frozen
`cv15_ringdown_spectra.json` A/B and `cv15_mode_pair_ratio_band.json`) is NOT
regenerated — regenerating it would destroy the A/B it exists to preserve, and
the #740 realization it records is unreachable once `two_plane` is a
`TypeError`. Append:

> ISSUE #931 (2026-09-07): the blindness the paragraph above states is a
> property of the f0 GATE and is unchanged. What changed is that the
> realization it was blind to is no longer reachable — `two_plane` is deleted,
> a conductor's plane is a declaration, and `assert_realized_stack` gates the
> realized planes against it (including the absence of a wall at k_patch+1).
> The frozen #740 A/B stays committed as historical evidence and now rests on
> the preserved pre-#931 leg rather than on the live one.

### 4d. `artifact_paths`

Add the preserved before-side leg, so the before/after pair is discoverable
from the manifest rather than only from a commit message:

```
"validation/crossval/_15_patch_results/rfx.json",
"validation/crossval/_15_patch_results/openems.json",
"validation/crossval/_15_patch_results/rfx_pre931_two_plane_ground_1f005d0d.json"
```

(`rfx_one_plane_ground_b29f9de7.json` is deliberately still not listed — it is
cited from `claim_scope` and from `validation/README.md`, which is where it is
looked up.)

## 5. What must NOT change

* No gate threshold in any of the four cases. cv15's f0 envelope stays 8 %,
  the settling bar -40 dB, passivity 1.05, the directivity envelope 3 dB;
  cv14's Gate 0 tolerance stays 1e-9 m and Gates 1-3 stay 1 % / 2 % / 0.1/T;
  cv16 stays 3.3 / 4.0 dB; cv17 stays 6.3 dB and 0.5 % relative.
* `role`, `evidence_levels`, `execution_tiers`, `expected_exit_codes` and
  `failure_sentinel` on all four.
* cv15's delegation of patch ACCURACY to case 05.
