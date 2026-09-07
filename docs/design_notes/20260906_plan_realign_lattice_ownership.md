# Lattice ownership contract for conductors — design + migration plan (2026-09-06)

Status: **DECIDED by the PI 2026-09-06** (issue #931). This note is normative for the
implementation branch `feat/931-lattice-ownership`. It amends the 2026-09-05 post-v1.8
plan's item 1 from "visualize the rasterization" to "define and enforce the ownership
contract"; the visualization work then reads the contract instead of re-deriving.

## 0. The defect in one paragraph

`Box.mask_on_coords` samples a Box half-open `[lo, hi)` at E-NODE coordinates and the
array is called `cell_mask`. `rfx.boundaries.pec.tangential_edge_masks` zeroes `E_a` at
entry `c` iff `c` is masked and a neighbour of `c` along axis `a` is masked (the #677
"thin-sheet neighbour rule"). Net effect, measured on synthetic slabs (issue #930/#931
tables): a PEC body is realized as a stack of sheets, one tangential wall per masked node
plane, at that plane. The far (`hi`) face is never a wall at any thickness; in-plane, the
`hi` row of a footprint is missing the same way. `two_plane` (#706) puts the far plane back
for `t = 1` only. #702 re-samples a 1-node sheet's "own cell" material at its live edge.
cv19 feeds its oracle `(L_c + 1)·dx`; cv15 draws the ground one cell below the substrate
floor and needs `two_plane` on one sheet and not on the other. Every one of those is a local
repair of the same undeclared semantics. The rule itself was a correct sheet rule; it was
never declared as one, and the primitive it was attached to is called `Box`.

## 1. The contract (normative)

A conductor is exactly one of three things, and the declaration says which.

| kind | what it is | declared by | E edge is PEC iff |
|---|---|---|---|
| **volume** | a set of primal cells | `sim.add(shape, material=<pec>)` — Box, Sphere, Cylinder | the edge is incident to an occupied cell |
| **sheet** | a footprint on ONE node plane, zero thickness | `sim.add_thin_conductor(shape, ...)` (PEC by default; lossy with `surface_impedance_f0`) | the edge lies in the plane and both of its end nodes are in the footprint |
| **wire** | a 1-D path of edges | `PolylineWire` | the edge lies on the path |

One sentence covers all three: **an E component is PEC iff its own location is inside the
closed conductor region.** The three rows are that sentence evaluated for a 3-D, 2-D and
1-D region on the Yee lattice.

### 1.1 Index conventions (unchanged, now written down)

* Node `i` sits at `x_i = (i − pad)·dx` (uniform) or at the cumulative edge position (NU).
  Primal cell `i` spans `[x_i, x_{i+1}]`; node `i` is its LOWER corner.
* `Ex[i,j,k]` is at `(x_{i+½}, y_j, z_k)`, `Ey[i,j,k]` at `(x_i, y_{j+½}, z_k)`,
  `Ez[i,j,k]` at `(x_i, y_j, z_{k+½})`.
* The occupancy array `C[i,j,k]` means **primal cell `(i,j,k)` is conductor**. For PEC
  VOLUMES it is sampled at **cell centres**, half-open: cell `i` is occupied iff
  `lo ≤ x_{i+½} < hi` (Box), or the centre lies inside the shape (Sphere, Cylinder). On node
  planes this gives exactly the cells the node sampler gives (`i0 .. i1−1`); off-lattice it
  rounds each face to the NEAREST plane instead of always outward (node-half-open + far
  face) or always inward (the old rule), and a corner drawn on a cell midpoint — the cv18
  "midpoint recipe" — lands on the plane it intended (lo inclusive, hi exclusive at the tie).
  A Sphere centred on a node realizes symmetric about that node (the old rule was one cell
  short on every `+` side). **Dielectric sampling is untouched** (node, half-open), so
  every dielectric-only fixture stays bit-identical; for node-aligned corners PEC and
  dielectric cells coincide by construction. NU grids: centres are `node + d/2` from the
  per-cell size arrays; a traced mesh keeps the traced path.

### 1.2 Volume realization

```
Mx[i,j,k] = C[i,j,k] | C[i,j-1,k] | C[i,j,k-1] | C[i,j-1,k-1]
My[i,j,k] = C[i,j,k] | C[i-1,j,k] | C[i,j,k-1] | C[i-1,j,k-1]
Mz[i,j,k] = C[i,j,k] | C[i-1,j,k] | C[i,j-1,k] | C[i-1,j-1,k]
```

The backward shifts use the #689 boundary convention already spelled in
`_axis_neighbors`: explicit zero pad on a non-periodic axis, wrap on a periodic axis and on
a length-1 axis. Consequences that follow, and that the contract test pins:

* a Box drawn `z_a → z_b` on node planes realizes tangential walls at BOTH `z_a` and `z_b`
  and shorts every normal edge between them; realized thickness = drawn thickness;
* a 1-cell PEC Box is a filled slab with two faces (what `two_plane` produced), at every
  thickness, on every axis, with no flag;
* a body touching the domain face at cell `n−1` has its far face on plane `n`, which the
  domain BC owns (same as today);
* the rule is invariant under mirror and axis permutation of the geometry.

### 1.3 Sheet realization

A sheet is `(normal_axis a, plane index k, footprint F)` where `k` is a **static integer**
and `F` is a boolean node mask on the plane. For a Box shape the footprint is sampled
**closed** `[lo, hi]` on the two in-plane axes (so the drawn rectangle is realized exactly,
including its `hi` row); for any other shape `F = shape.mask_on_coords(...)` restricted to
plane `k`. Realized edges: for `a = z`,

```
Mx[i,j,k] = F[i,j] & F[i+1,j]      (edge from node i to i+1 at row j)
My[i,j,k] = F[i,j] & F[i,j+1]
Mz                                  unchanged — normal E through a sheet stays live
```

and cyclically for `a = x, y`. A sheet owns NO cell: it adds nothing to `C`, writes no
`eps_r`/`sigma`, and is realized at exactly one plane. Plane selection for a declared
sheet: the node plane nearest the shape's mid-plane along `a`; an exact half-cell tie
resolves to the LOWER plane (today's `n_vol == 1` rule for a face-registered 1-cell Box,
kept so existing declarations land where they land today). `fidelity_report` and preflight
print the realized plane in physical units.

Footprints of all sheets on the same `(a, k)` plane are UNIONED before the edge rule is
applied, so two abutting sheets realize seamlessly (per-sheet application would leave the
shared edge live — a slit). Sheets on adjacent planes stay two films with a live normal
edge between them (#690 semantics). A zero-thickness Box (`lo == hi` on the normal axis)
is the canonical way to say "sheet on this plane". A sheet whose declared mid-plane sits on
an exact half-cell tie (a face-registered one-cell Box) gets a preflight WARNING naming
both candidate planes and the one chosen. On a length-1 axis (2-D lane) a sheet whose
normal is that axis has no tangential components; it is realized as a 2-D volume of its
footprint cells, with a notice.

A lossy (`surface_impedance_f0`) sheet uses the SAME footprint and the SAME edge set; the
#677 G4 identity ("f0 toggles loss, never geometry") is then true by construction, not by
a test that compares two rules.

### 1.4 Wire realization

`PolylineWire` with `radius ≥ ½` local cell is a volume (centre-sampled tube). Below that
it is a filament: the E edges of the axis-aligned lattice path joining the nearest nodes of
consecutive vertices. Only axis-aligned segments are supported in this change (a diagonal
segment raises; today such a wire silently rasterizes to disconnected nodes and realizes
nothing). No crossval or example uses `PolylineWire`; four tests do.

### 1.5 What `sim.add(Box, material=pec)` refuses

* A PEC Box with **exactly one zero-extent axis** (`lo == hi` there) IS a sheet
  declaration — zero thickness is a statement of intent, not an inference — and is realized
  exactly as `add_thin_conductor` would realize the same Box (plane = nearest node to the
  declared plane, tie → lower; an off-node plane is reported as a NOTICE with its offset).
  This keeps the documented five-line patch example and `first-patch.mdx` valid. Two or
  three zero-extent axes raise (a line or a point is not a conductor; use `PolylineWire`).
* A PEC Box with `0 < extent < one local cell` along any axis raises
  `ValueError: ... a Box is a volume; declare a sheet (a zero-thickness Box or
  add_thin_conductor) or resolve the thickness`. Nothing is inferred from raster thickness
  or drawing direction.
* A PEC Sphere / Cylinder / Box that rasterizes to ZERO cells (a via thinner than ~0.7 cell,
  a post between cell centres) raises, naming `PolylineWire` for a filament and the minimum
  radius for a volume — the #369 silently-vaporized-metal class, now an error.
* `add_thin_conductor` with a shape thicker than one local cell along its normal raises
  ("not a sheet; use add() for a volume").
* `two_plane` is gone. Passing it is a `TypeError` (unknown keyword), not a deprecation.
* There are no per-entry realization knobs. A test greps `rfx/` for `two_plane` and for any
  `realization=` style keyword on geometry entries and fails on a hit.

### 1.6 Soft (differentiable) path

`apply_pec_occupancy(state, occ, periodic)` uses the noisy-OR of the four incident cells,
`M = 1 − Π(1 − o_c)`, with the same #689 shifts. At binary occupancy it is bit-identical to
1.2 (pinned on a shape battery that includes non-periodic seams). Sheets enter the soft
path as static masks OR'd in (their plane is not a traced quantity — the `argmin` cliff).
`rfx/topology.py` gets no sheet or two_plane logic.

### 1.7 One source, every consumer

`rfx.boundaries.pec.realized_pec_edge_masks(occupancy, sheets, periodic) → (Mx, My, Mz)`
is the only function that turns geometry into PEC edges. Consumers that today re-derive
from `pec_mask` switch to it or to the two helpers built on it:

* `realized_wall_planes(axis, region=None)` — sorted node-plane indices where a tangential
  wall exists (preflight #703/#767 cavity checks, #729 declared-vs-realized, cv15
  `assert_realized_stack`, oracles);
* `edge_is_pec(component, i, j, k)` — wire-port live-cell logic (#556 end-gap, #929
  `port_in_pec`), probes, sources.

Consumers: `simulation.py` / `nonuniform.py` step functions, `runners/distributed_nu.py`
shmap twins (the three edge masks are sharded along x like the cell mask — a PEC sheet must
not vanish on the distributed lane), `visualize.py`, `fidelity.py`, `_preflight.py`,
`sources/sources.py` (`_wire_port_live_cells`), `sources/coaxial_port.py`,
`probes/probes.py`, `probes/sparam_driver.py`, `probes/msl_wave_decomp.py` (finds the MSL
trace by scanning a `pec_mask` column above the substrate — a sheet trace is not in
`pec_mask`, so it must read `realized_wall_planes`), `api/_execute.py` (2-D lane
`pec_mask[:, :, 0]`), `api/_sparams.py`, `materials/thin_conductor.py`
(`build_sheet_impedance_ctx`), `vmap_sweep.py`, `interop/_design.py` (an IR document
carrying `two_plane` is rejected with a message, not ignored).

### 1.8 Scope fences (each is a decision, not an omission)

* **Domain-boundary PEC** (`BoundarySpec` faces, `apply_pec` / `apply_pec_faces`) is NOT a
  body and keeps its convention (E_tan = 0 on the face plane at index 0 / N). cv09, cv10,
  cv14, cv24, `adi_solver_demo`, `hello_world`, `resonance_harminv` are the controls: they
  must not move.
* **Sigma-fill conductors** — `rasterize(..., sigma=1e7)` in cv16 / `rcs_scattering`,
  `stamp_coaxial_line`'s shell and pin, the DC-fold `add_thin_conductor` (`sigma_bulk <
  1e6`, no `f0`) — are a LOSSY VOLUME model (fields decay inside a conductive cell); they
  are not PEC realization and are unchanged here. Their own node-vs-cell debts (the coax
  shell one cell inside `b`, `r_os = b + 2dx`) get a follow-up issue. A test pins that the
  two models are not silently equated.
* **Kottke Stage-2 (`subpixel_smoothing='kottke_pec'`) and Dey–Mittra Stage-1 conformal**
  paths are subpixel models with their own interior selection; unchanged. Their binary
  parts are a follow-up (they already realize both faces for node-aligned Boxes).
* **Dielectric sampling** (node, half-open) and the `DesignRegion` / `eps_override` index
  mapping (`optimize.py`, `topology.py`, inverse-design examples) are unchanged. The design
  region's inclusive `+1` is a separate, documented debt (#729 class), not touched here.
* **`forward(pec_mask_override=)`** is a VOLUME override (cells) and stays one.

### 1.9 Consumers that today scan `pec_mask` for metal (all switch to §1.7)

* wire-port live/dead (`_wire_port_live_cells`): live iff the port component's edge at that
  index is not in `(Mx, My, Mz)`;
* port "clearing" (wire live cells, lumped cell, MSL cross-section cells): today sets
  `pec_mask[c] = False`; becomes `clear_edges(edge_masks, cells)` — the three E entries at
  those indices are un-zeroed. Same for `pec_occupancy` clearing;
* MSL trace detection (`probes/msl_wave_decomp.py`, `_preflight._msl_realized_substrate`):
  read `realized_wall_planes` on the column;
* waveguide guide width (`_preflight._port_transverse_spans`): distance between realized
  wall planes (today measures a 40 mm guide as 42 mm — the #868 class);
* two-run S-matrix reference (`strip_interior_pec`): strips sheets too, or a sheet iris
  reads S11 = 0;
* reference-plane conductor footprint (`_execute._refplane_conductor_mask`, `refplane.py`),
  `conductor_mask()` / `conductor_footprint`: cells of volumes ∪ footprints of sheets;
* `auto_configure` z-feature detection: sheet planes are features;
* the YAML front-end (`rfx/config/loader.py`): gains a sheet entry (`thin_conductor`);
* `interop/_design.py` + IR schema: `two_plane` (currently a REQUIRED boolean) is removed,
  sheets are added, IR version bumped; a document carrying `two_plane` is refused.

## 2. Deleted

| what | where | why |
|---|---|---|
| `two_plane` kwarg, `_GeometryEntry.two_plane`, IR field, `_two_plane_cell_mask`, `_refuse_two_plane`, `two_plane_extension_masks`, `_place_at_next_plane`, ctx fields on every lane | `api/__init__.py`, `api/_spec.py`, `api/_execute.py`, `boundaries/pec.py`, `simulation.py`, `nonuniform.py`, `runners/*`, `vmap_sweep.py`, `interop/_design.py`, `probes/refplane.py`, `fidelity.py`, `visualize.py`, IR schema, Studio | t=1-only patch; subsumed by 1.2 |
| `resample_sheet_node_materials`, `sheet_normal_live_axis_masks`, `_subcell_box_axis_window`, `_statics_on_coords`, `collect_thin_conductor_sheet_inputs` | `geometry/rasterize_grid.py` and its callers in `api/_compile.py`, `nonuniform.py` | a sheet owns no cell, so there is no "own cell" to re-sample. The one physical case it served — a stack-up that leaves a slot for the foil — becomes a preflight notice (§3) |
| `tests/locks/test_two_plane_pec_slab.py`, `tests/unit/materials/test_sheet_node_permittivity.py` (resample tests) | tests | pin deleted mechanics; replaced by the contract tests in §5 |
| oracle-side compensations: cv19 `(L_c + 1)·dx`, `t_c = round(t/dx) + 1`, `L_c = round(L/dx) − 1`; cv18/csg "midpoint recipe" as a requirement; cv15 `Box(z_sub_lo − DX → z_sub_lo)` ground | crossval scripts + fixtures | under 1.2 drawn = realized, so the oracle takes the drawn value |

The full per-site list (every crossval, example, test, doc) is the inventory in §4.

## 3. Added

* `realized_pec_edge_masks`, `realized_wall_planes`, `edge_is_pec` (§1.7).
* `rasterize_geometry` returns `sheets: list[SheetSpec]` beside `pec_mask`; PEC thin
  conductors go to `sheets`, never to `pec_mask`.
* Preflight (input-fidelity only, per `feedback_preflight_input_fidelity_only`):
  * `pec_box_subcell` — ERROR: the §1.5 refusal, with the physical thickness and local cell;
  * `pec_box_one_cell` — WARNING: "PEC Box '<name>' is one cell thick along z: realized as a
    filled slab with walls at z = … and z = …. If this is foil, declare it with
    `add_thin_conductor`" (every crossval foil today is drawn this way, so this fires on
    unmigrated scripts by design);
  * `pec_zero_cells` — ERROR: the §1.5 zero-cell refusal (also reported here for shapes the
    rasterizer clipped to nothing);
  * `sheet_plane_realized` — NOTICE per sheet: declared mid-plane, realized plane, offset;
    WARNING when the mid-plane is an exact half-cell tie (both candidate planes named);
  * `sheet_slot_vacuum` — ERROR-grade WARNING: the node plane of a sheet carries vacuum while
    both neighbouring cells carry dielectric (a stack-up drawn with a slot for the foil);
    the one E edge that reads that node sits half a cell into the dielectric, so the cavity
    gains a vacuum cell in series (the #702 measurement: 17 % on a 127 µm stack). Remedy:
    extend the dielectric boxes to the sheet plane. Nothing is re-sampled silently;
  * every existing PEC/cavity/port validator reads `realized_wall_planes` / `edge_is_pec`.
* `fidelity_report`: per PEC entry, drawn extent vs realized wall planes per axis, in input
  units.
* `CHANGELOG`: this is a breaking API change (a public kwarg removed, a realization
  change). It ships as **2.0.0** under the #825 umbrella, not as a 1.x minor.

## 4. Migration (filled from the exhaustive inventory)

See `docs/design_notes/20260906_lattice_ownership_inventory.md` (generated from the
18-reader inventory run; every PEC site in `rfx/`, `validation/`, `examples/`, `tests/`,
`docs/`, `scripts/` with intent, thickness, downstream artifacts, action, recompute cost).

Migration rules, in priority order:

1. **Foil drawn as a 1-cell PEC Box** (ground planes, patches, traces) → `add_thin_conductor`
   with the SAME physical corners. Realized plane = nearest node to the mid-plane (tie →
   lower), which for a face-registered 1-cell Box is the plane it lands on today.
2. **Foil drawn one cell OUTSIDE its interface** to park the wall on the interface (cv15
   ground) → draw it AT the interface as a sheet; delete the `two_plane` flag.
3. **Walls, irises, posts, plates drawn as volumes** → unchanged drawing; the realization
   gains its far face. Every number derived from today's realization (oracle inputs,
   fixture values, locks, gate bounds) is recomputed from the drawn geometry, and the
   compensation that produced it is deleted, not re-tuned.
4. **Tests pinning the old mechanics** → rewritten against §1, or deleted when the mechanic
   no longer exists.
5. **Docs** stating the old rule → rewritten from §1.

## 5. Verification (pre-declared)

Contract tests (`tests/contracts/test_lattice_ownership_contract.py`):

* slab battery: Box `t = 1, 2, 3` on each axis, each realizes walls at `lo` AND `hi` and
  no live normal edge inside; a sheet realizes one plane and a live normal edge;
* footprint battery: a patch Box realizes its drawn rectangle exactly (closed), as volume
  and as sheet;
* mirror / axis-permutation invariance of the realized edge set;
* soft ≡ hard at binary occupancy on the battery, including bodies on faces 0 and `n−1` of a
  non-periodic axis and the periodic seam;
* PEC thin conductor ≡ sub-cell sheet ≡ f0 sheet footprint (G4, by construction);
* `grep two_plane rfx/` = 0; no realization keyword on geometry entries;
* distributed-NU shmap parity with the single-device lane on the battery.

Physics falsifiers, pre-declared before any recompute:

* **cv19 (WR-90 iris filter)**: the fixture's cavity leg goes from `(L_c + 1)·dx` to the
  drawn `L_c·dx` and the iris-thickness offset from `−0.68` to `≈ +0.32` cell. If the
  recomputed residual does not move by about one cell, the diagnosis is wrong and the
  volume rule is not merged.
* **cv15 (RT5880 patch)**: `assert_realized_stack` passes with NO flag: walls at `z_sub_lo`
  and `z_sub_hi`, four substrate cells between. Realized in-plane patch = drawn.
* **One-cell volume witness (new)**: cv18's iris-thickness sweep gains `t_c = 1`; the
  mode-matching oracle at `t = dx` must sit on the same residual curve as `t_c = 2..8`. Today
  no independent witness says the two-wall rule is right at one cell.
* **cv16 (PEC sphere Mie)**: recomputed under centre sampling; the realized sphere is now
  symmetric about its centre. Expected to improve; if #820's translation variance moves,
  record it, do not claim it.
* **Dielectric-only cases** (cv04, cv17, cv22, cv23 and every example without a conductor
  body): bit-identical results before/after — the change must not touch them.
* **Multilayer board A/B (memory 2026-08-28, VESSL 369367256724)**: the old `two_plane` arm
  correlated +0.265 with CST, one-plane +0.829. Under the contract the foils are sheets, so
  the board must stay at the one-plane figure; a drop below 0.8 blocks the default.

R3 for every commit on the branch: `R3: memory=rfx-known-issues.md("top face plane is never
zeroed" 2026-08-28; two_plane A/B verdict) | R2-attempts=0 (redesign, not a repeat) |
falsifier=<the §5 item exercised>`.
