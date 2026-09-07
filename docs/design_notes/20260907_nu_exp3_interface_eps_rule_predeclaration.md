# E3 — the interface-node eps rule in the PRODUCTION path (opt-in prototype for #931) — pre-declaration

**Status:** pre-declaration. This commit precedes the source change
(`Simulation(..., interface_eps=...)`, `rfx/runners/nonuniform.py`), the
instrument wiring (`validation/research/multiband_nu/w7_accuracy_ad.py
--interface-eps`), the tests and every measurement. Every numeric window
below is frozen from this commit onward; results are appended under
"Results", never edited into the windows. One attempt per arm; a fired
falsifier is a result, not a bug to tune away.
**Class:** observable-vs-analytic on the lane F A1 z-stratified cavity
(family: W7 A1 `w7_accuracy_ad`, note
`20260907_nu_band_accuracy_ad_predeclaration.md`, whose oracle, exact
discrete model, harness, extraction, gates and 'dual' rows are reused, not
re-derived), plus a bit-identity contract on the default path.
**Tree:** worktree `rfx-nu-exp3`, branch `exp/nu-experiments-e3`, based on
`feat/nu-band-accuracy-ad @ fea9f078` (which carries `feat/nu-band-profile`
and `origin/main d990e18c`). E2 (`exp/nu-experiments-e2`, worktree
`rfx-nu-exp2`) is read, cited, and not modified.
**Import provenance for every number here:** `rfx.__file__` printed as
`/Users/byungkwankim/Documents/rfx-nu-exp3/rfx/__init__.py` (PYTHONPATH
pinned to this tree; the editable install points at the primary checkout).
Every JSON the instrument writes records `rfx.__file__`, `git_sha`,
`git_dirty`, `argv` and `started_utc` (the W6/W7/E2 convention).
CPU only (a GPU lane runs elsewhere; the machine is shared).
Date: 2026-09-07 (KST).

**This is a PROTOTYPE for #931 (geometry -> lattice ownership).** It adds
an opt-in flag and measures what the flag does on one fixture family. It
changes no support-matrix row, no default, and no committed number. Whether
the rule becomes a default, and for which lanes, is #931's decision, not
this note's.

## 0. Scope — the question

Lane F (section 1a/1b of its note) found that the production NU material
assembly (`rfx.runners.nonuniform.assemble_materials_nu`) samples each E
node's material by the half-open `[lo, hi)` rule on the node coordinate.
At a node-aligned interface the node's material is decided by the last ulp
of the running cell sum (nine meshes, three assignment patterns), and for a
tangential E component either one-sided choice is a first-order error:
+0.15 % at s = 1 on the uniform A1 mesh, fitted order 1.44 (UC) / 1.78 (MB)
against 2.007 / 2.010 for the dual-cell-average rule the instrument
assembles for itself. E2 measured the law behind the average
(`e_shift = K t (h - t)`, second order at fixed fill) and stated the
normal-E rule without measuring it.

The question here: realise the second-order rule INSIDE the production
assembly, behind an opt-in switch, and measure that the production path
then reaches the instrument's numbers. Two claims, both falsifiable:

1. **Bit-identity of the default.** `interface_eps="sampled"` (the default)
   changes nothing: every existing test in the four batteries and the
   example-fidelity contract passes unchanged, and an explicit
   `interface_eps="sampled"` run is bit-identical to a run without the
   kwarg.
2. **The production path realises the rule.** With
   `interface_eps="dual_average"` the w7 A1 'production' arm reaches the
   'dual' arm's numbers: fitted order >= 1.8 on UC and MB, and the same
   per-scale errors to <= 0.5 MHz.

## 1. Findings made while writing this note (zero FDTD)

### 1a. Where each E component sits, and what the rule therefore is

`materials.eps_r` is ONE co-located scalar field on the node lattice,
consumed by all three E components (`update_e_nu`); per-component eps
exists only on the `aniso_eps` channel (`update_e_nu_aniso`, fed today by
`compute_smoothed_eps_nonuniform` when `subpixel_smoothing=True`). The
stagger the NU update realises (read from `rfx/core/yee.py` and
`rfx/nonuniform.py`, not assumed): the E update differences H BACKWARD
with the dual spacing `inv_d_e[k] = 2/(d[k-1]+d[k])`, the H update
differences E FORWARD with the primal spacing `1/d[k]`. So E_c(i, j, k) is
the edge running from node (i, j, k) to the next node along c: it lies
INSIDE cell index k_c along its own axis (half-open, the cell whose low
corner is the node) and ON the node lines of the other two axes, bordering
the four cells `(i_a - 1 | i_a) x (i_b - 1 | i_b)` there.

The rule that follows, per component, from the dual-cell Ampere balance
(E2 1a: E tangential is continuous across an interface, so the dual-cell
integral of eps E is E times the volume integral of eps):

- **tangential axes** (the two axes `a, b != c`): E_c takes the
  area-weighted ARITHMETIC mean of the four cells sharing its edge,
  weights `d_a[i_a - 1 | i_a] * d_b[i_b - 1 | i_b]`. On a z-stratified stack
  (all cells equal in x and y) this reduces exactly to the instrument's
  `dual_eps_nodes`: `(eps[k-1] d[k-1] + eps[k] d[k]) / (d[k-1] + d[k])`;
- **its own axis** (the normal component of an interface normal to c):
  E_c takes the eps of the ONE cell its edge lies in, `eps_cell[k_c]`. A
  node-aligned interface normal to c does not cross the E_c edge at all —
  the edge starts on the interface and runs into cell k_c — so there is
  nothing to average, and the harmonic (D-continuous) rule E2 1a/1c states
  for a normal component reduces to the single cell's value when the edge
  is inside one material. The harmonic rule is needed only when an
  interface cuts the edge (a sub-cell layer, E2's S1 family on Ez) — not
  this lane's case and not measured here;
- at the first node of an axis there is no lower cell: one-sided (the
  cell's own eps); the trailing bounding node's phantom cell copies the
  last real cell (so the last node is one-sided too). This is exactly the
  instrument's end-node convention (`out[0] = ce[0]`, `out[n] = ce[-1]`).

The cell eps is sampled at the CELL CENTRE (`node + d/2`, exact float64
spine) by the same `rasterize_geometry` the node path uses, so no cell
centre ever lands on a node-aligned interface and the ulp residue of lane
F 1a cannot enter. An interface strictly inside a cell (sub-cell feature)
is staircased at the centre — E2's S0 class, first order; stated, not
measured.

Consequence for the SAMPLED default, recorded: at a node whose two
adjacent cells differ along axis a, the sampled value is right for the
normal component E_a when the ulp landed "upper" (cell k's material) and
wrong for it when it landed "lower"; it is one-sided (first order) for
both tangential components either way. The opt-in rule fixes all three.

### 1b. Why the rule lives on the per-component channel and what the scalar field keeps

The smallest change that gives the tangential components the mean WITHOUT
giving the normal component a smeared value is to emit three arrays on the
existing `aniso_eps` channel and leave `MaterialArrays.eps_r` (the scalar
field) as the sampled column. Putting the mean into the scalar field would
give E_z at a substrate/air node `(4.3 + 1)/2` where the E_z edge sits in
air — a first-order error of the S0 class on exactly the component that
dominates microstrip-class models. Physical fidelity first: the normal
component keeps its cell. So under the opt-in:

- the three E components use `(eps_x, eps_y, eps_z)` from the rule
  (`update_e_nu_aniso`);
- `materials.eps_r` stays sampled and is what source normalisation
  (`make_current_source`), the CPML coefficients (`apply_cpml_e`), the
  fidelity report's materialisation rows and `rfx.visualize` read — the
  same split `subpixel_smoothing=True` already has on this lane;
- `materials.sigma` stays sampled and isotropic (`update_e_nu_aniso`
  applies sigma isotropically, as documented there). A lossy interface is
  outside this prototype's domain.

### 1c. What the prototype refuses (loud, never silent)

`interface_eps="dual_average"` raises `ValueError` / `NotImplementedError`
at run time when combined with any of: Debye/Lorentz materials (the
dispersive scan branch does not consume `aniso_eps`); `subpixel_smoothing`
(two eps rules); thin conductors (sheet resample and the sigma folds are
scalar-field operations, `resample_sheet_node_materials`); lumped RLC
(folds into `eps_r`); `eps_override` (the AD-material channel replaces the
scalar field only); a traced (mesh-as-design-variable) profile (the
cell-centre rasterization is host float64); the distributed NU lane and
the S-parameter NU lane (neither carries `aniso_eps`). The surface-
impedance sheet + `aniso_eps` refusal (#677 v1) already exists and holds.
Every refusal is a test (section 4), not a measurement.

## 2. Fixture — lane F A1, verbatim

The z-stratified PEC cavity of lane F 2.1: core 4.3 | thin 3.0 | core 4.3
| air, interfaces at z = 14, 16, 30 mm, L_z 44 mm, a = 30 mm, b = 3 mm, LSE
m = 1, p = 5, `f_true = 10 561 719 600.896 Hz` (transfer-matrix oracle,
selfcheck (i)/(i')/(i'') to 1e-12). Meshes UC / MB at s = 0.5, 1, 2
(transverse cell 0.25 s mm; UC uniform 0.5 s mm; MB builder bands cap 1.4),
AZ at the same scales for the column check only. Source/probe/waveform/
extraction: lane F 2.1 (15 ns harminv, 10 ns truncation invariance).

Under the opt-in the w7 'production' arm reads the E_y component of the
rule's output at the central (i, j) column instead of the scalar field,
then runs on the instrument harness as before (Ey only; the model of lane
F 3.1 applies unchanged to that column). Nothing else in the arm changes.

Lane F's measured 'dual' rows, the reference for E3-F2 (MHz, `err_hz`
from `results/w7_accuracy_ad.json`, invariance-passing, all valid):

| arm | s = 0.5 | s = 1 | s = 2 |
|---|---|---|---|
| UC dual | -3.9786 | -16.0400 | -64.3221 |
| MB dual | -7.5005 | -30.0934 | -121.6770 |

and its fitted 'dual' orders 2.007 (UC) / 2.010 (MB); 'production'
(sampled) orders 1.438 / 1.780.

## 3. Falsifiers (frozen; tolerances never widened after measurement)

Every window is evaluated by the instrument's own judges
(`judge_a1`: G3 model residual, invariance, `production_orders`) or by
the replay test; none is computed by hand.

- **E3-B (bit-identity of the default).** (a) The batteries
  `tests/unit/nonuniform`, `tests/unit/geometry`, `tests/unit/materials`,
  `tests/oracle` with `-m "not gpu and not slow"` and
  `tests/contracts/test_example_fidelity_contract.py` pass on the changed
  tree with the same pass/skip counts as on `fea9f078` (recorded both
  ways). (b) A 200-step NU run (A1 MB s = 2 mesh, PEC, Ey source) with
  `interface_eps="sampled"` explicit and with the kwarg omitted: time
  series bit-identical, `max |diff| = 0` exactly; and
  `assemble_materials_nu` output arrays bit-identical. (c) The lane F
  section 1a interface table (nine meshes, three nodes each) is
  reproduced by the default path exactly (w7 `--selfcheck`, unchanged
  constants). Window: zero differences, zero test regressions.
- **E3-C (the production rule equals the instrument's rule).** For the
  nine (arm in uc, mb, az) x (s in 0.5, 1, 2) meshes, the E_y column the
  production rule emits at (nx//2, ny//2) equals
  `f32(dual_eps_nodes(prof))` at every node to `<= 5e-7` relative (one
  float32 ulp; the f64 arithmetic is the same formula, the x/y averaging
  is the identity on a transversely uniform stack), and the E_z column
  equals the cell-centre (half-open at the centre) eps of cell k at every
  node exactly. Window: `<= 5e-7` relative on E_y, `0` on E_z.
- **E3-O (oracle and selfcheck).** w7 `--selfcheck` `all_pass` on this
  tree with the default rule: oracle (i)/(i')/(i'') `<= 1e-12`; all 30
  model rows to `1e-7`; orders/ratios to `1e-3`; interface tables
  reproduced. Window: `all_pass = True`, else no arm runs.
- **E3-G3 (model residual).** Every measured unit:
  `|f_meas - f_model| <= 0.15 MHz` with f_model from the emitted column
  (lane F G3, same constant). Window: `<= 0.15 MHz`.
- **E3-V (run-length invariance).** `|f_meas(15 ns) - f_meas(10 ns)|
  <= 0.1 MHz` per unit, else the unit is INCONCLUSIVE and excluded from
  the fit (lane F rule). Window: `<= 0.1 MHz`; `>= 3` fit points per arm
  required for E3-F1, else INCONCLUSIVE.
- **E3-F1 (order).** Fitted order of |err| vs h on the production
  dual_average ladder, three scales, UC and MB: `p_uc >= 1.8` and
  `p_mb >= 1.8` (task window). Lane F's dual arm gave 2.007 / 2.010 and
  its sampled production arm 1.438 / 1.780; the exact discrete model
  predicts 2.004 / 2.009 for the dual column. Reported alongside: the
  lane F oracle-validity window `[1.8, 2.2]` on UC (a UC order above 2.2
  would be an anomaly, not a pass).
- **E3-F2 (per-scale agreement with lane F 'dual').** For each of the
  six (arm, s) units: `|err_hz(E3 production dual_average) -
  err_hz(lane F dual)| <= 0.5 MHz` against the table of section 2. Same
  rule realised in production vs in the instrument, same mesh, same dt,
  same waveform; the only differences are the column's assembly route
  and CPU-vs-lane-F float32 reduction order.
- **E3-R (refusals).** Each combination of section 1c raises before any
  step is taken; the error text names `interface_eps`. Pure tests.
- **E3-P (the report states the rule).** `fidelity_report()` prints
  `interface eps rule (NU lane): ...` on every NU model and the returned
  domain row carries `interface_eps_rule` ONLY under the opt-in (so the
  example-fidelity snapshot is untouched by the default). Pure test.

## 4. Declared commands, outputs, tests

```
cd /Users/byungkwankim/Documents/rfx-nu-exp3
export PYTHONPATH=/Users/byungkwankim/Documents/rfx-nu-exp3
PY=/Users/byungkwankim/Documents/rfx/.venv/bin/python
$PY -c "import rfx; print(rfx.__file__)"          # must be this tree
# E3-B(a): batteries, before (fea9f078) and after, counts recorded in Results
$PY -m pytest tests/unit/nonuniform tests/unit/geometry tests/unit/materials tests/oracle \
    -q -o addopts="" -m "not gpu and not slow" -p no:cacheprovider
$PY -m pytest tests/contracts/test_example_fidelity_contract.py -q -o addopts="" -m "not gpu"
# E3-O, then the A1 ladder with the opt-in on (one attempt per unit, its own JSON)
$PY -m validation.research.multiband_nu.w7_accuracy_ad --selfcheck \
    --out validation/research/multiband_nu/results/e3_interface_eps_rule.json
$PY -m validation.research.multiband_nu.w7_accuracy_ad --arms a1 --a1-arms uc,mb \
    --scales 2,1,0.5 --rules production --interface-eps dual_average \
    --out validation/research/multiband_nu/results/e3_interface_eps_rule.json
# E3-B(b,c), E3-C, E3-R, E3-P and the order replay
$PY -m pytest tests/unit/nonuniform/test_interface_eps_rule.py -q -o addopts=""
```

Outputs: `validation/research/multiband_nu/results/e3_interface_eps_rule.json`
(the w7 layout: `runs[]`, `selfcheck`, `a1.units` keyed `arm|s|production`
with `interface_eps_rule` recorded per row, `a1.judge`). Tests:
`tests/unit/nonuniform/test_interface_eps_rule.py` — the JSON-free
contract tests (E3-B(b,c), E3-C, E3-R, E3-P) and the replay (skips while
the JSON is absent; re-fits E3-F1 from the JSON's rows through
`w7.fit_line`, re-checks E3-F2 against `results/w7_accuracy_ad.json`'s
dual rows, E3-G3 and E3-V through the rows' own gates, and pins that every
row's `rfx_file` is this tree and `interface_eps_rule == "dual_average"`).

Source surface (declared before it is written): `Simulation.__init__`
gains `interface_eps: str = "sampled"` (validated against
`("sampled", "dual_average")`, stored as `_interface_eps`);
`rfx/runners/nonuniform.py` gains `INTERFACE_EPS_RULES` and
`assemble_interface_eps_nu(sim, grid, materials) -> (eps_x, eps_y, eps_z)`,
and `run_nonuniform_path` feeds it to the existing `aniso_eps` channel
after the subpixel block; the distributed NU lane (`_execute.py`) and the
S-parameter NU lane (`_sparams.py`) refuse the opt-in; `rfx/fidelity.py`
states the rule. `assemble_materials_nu`, `rasterize_geometry`,
`Box.mask_on_coords`, `coords_from_nonuniform_grid` are not edited.

## 5. Impact-sweep rule and the #931 note

No default moves. If E3-B fires (any count changes, any bit differs) that
is a STOP: the prototype is withdrawn from the branch until the difference
is explained in this note. If E3-C fires the rule is mis-assembled and the
ladder is NOT run (E3-C is checked by the test before the ladder). If
E3-F1 or E3-F2 fires with E3-C held, the difference between "same column
in the instrument" and "same column in production" is the result, and it
is reported with numbers.

For #931: this lane hands over (1) a working per-component realisation of
the tangential-mean / normal-own-cell rule on the NU lane behind a flag,
(2) the measured order and per-scale agreement, (3) the list of paths the
rule does not reach (1c) — which is the list #931 has to decide about
before any default changes. The uniform lane (`rfx/api/_compile.py`
`_build_materials`) has the same node sampling and the same ulp residue
and is NOT touched here.

## 6. Where this note departs from the task as written, and why

- The task named `interface_eps='sampled' | 'dual_average'` on
  `Simulation`; taken as written.
- "Arithmetic mean of the two adjacent cells' eps for the tangential-E
  components" is realised as the area-weighted mean over the FOUR cells
  sharing the edge (which is the two-cell mean whenever the interface is
  planar and the transverse cells match, i.e. on every fixture measured
  here); the four-cell form is the one the dual-cell balance gives in
  3-D and costs nothing extra.
- The normal component is not left at the sampled value: it takes the
  cell its edge lies in (1a). That is a change of the normal component at
  nodes where the ulp landed "lower" — deliberate, stated, and checked by
  E3-C's E_z clause.
