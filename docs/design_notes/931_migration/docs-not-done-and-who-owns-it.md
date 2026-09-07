# #931 Docs group — what it did NOT touch, and who owns each

The Docs group's brief scoped its `scripts/` work to the sites that **pass
`two_plane`** or **monkeypatch the #702 resample**. Six files matched and all
six are migrated or retired. The rows below are the rest of what the
docs-scripts inventory found under `scripts/` and in code, left deliberately.
Each is a real migration; none is done.

## `scripts/diagnostics` families that draw foil as one-cell PEC Boxes

None of these passes `two_plane`, so none is broken at HEAD — they build, and
their conductors simply gain their far face. That is exactly why they need a
re-run rather than a rescue: every number they published was measured on the
old realization.

| family | what it draws | who should own it |
|---|---|---|
| `patch_edgefed_*` (13 files) + `patch_inset_match_vs_pull`, `patch_uniform_fine_fr4_witness` | ground / MSL trace / patch as ONE-cell PEC Boxes on the shared RO4003C (h = 0.787 mm, dx = h/4) and RO4350B (h = 254 µm) fixtures; vias stay volumes | the patch lane — crossval-A / crossval-C, whichever re-solves the edge-fed board |
| `msl_*` (14 files: `msl_z0_bias_floor_sweep`, `..._realized_anchor`, `msl_thru_mesh_convergence`, `msl_thru_meshconv`, `msl_vi_flux_oracle`, `msl_passive_port_reflection`, `msl_probe_clearance_bias`, `msl_beta_rail_e2e`, `build_msl_notch_rfx_dx50`, `build_msl_thru_phase_dx50um_reference`, `compare_msl_thru_openems_reference`, `run_msl_broad_e5_sweep`, `graded_z_lowz_demo`, `thru_singular_value_dx_ladder`) | substrate 0..H_SUB with a one-cell PEC trace on top, ground from a `BoundarySpec` face | crossval-B (cv06b, cv07) and crossval-E (cv20) — the traces are the same board |
| `wr90_port/*` (8 files) | the PEC short as a 2-cell `Box` at x = 145 mm on a 1 mm grid, node-aligned, full transverse cross-section | crossval-B / the waveguide lane. **Do not edit the geometry** — re-run `boundary_cell_trim` (≈10 s) and `pec_short_position_sweep` as pre/post controls and record that \|S11\| is unchanged. A short reflects at its FRONT wall either way; if \|S11\| moves, the front-face reasoning is wrong and that is a STOP |
| `i544_n_live_advisory_vs_assembler`, `thru_singular_value_dx_ladder`, `i517_mixed_solve_vs_ratio_measurement` | positionally unpack `Simulation._assemble_materials(grid)` for `pec_mask`, then `_wire_port_live_cells(grid, wp, pec_mask)` | preflight / core — these are the §1.9 wire-port live-cell consumers and should read `edge_is_pec`. `i544_n_live_advisory_vs_assembler` is the natural acceptance witness for that change |
| `build_rcs_mie_reference` | models the PEC sphere as a **sigma fill** (`rasterize(grid, [(sphere, 1.0, 1e7)])`), no `pec_mask` | crossval-C, with cv16. Design note §1.8 fences the sigma path, so the decision to record is whether the Mie fixture is regenerated on the PEC path or stays a sigma-fill reference — do not let the two diverge unannounced |
| `precompute_gallery_artifacts`, `_gallery_v3_patch_figs`, `patch_edgefed_s11_validation`, `msl_flux_ratio_dof`, `capture_msl_replay_fixture`, `_wr90_port_oracle_matrix`, `harnesses/v173a_physics_equivalence` | top-level `scripts/` builders; `precompute_gallery_artifacts` draws the ground ONE CELL BELOW `z0_gnd` — the migration-rule-2 compensation — and feeds `docs/public/gallery/assets` | no group was assigned top-level `scripts/`. The gallery assets are a dated record (rfx 1.6.5, commit 1eb551b) and are not re-rendered, but the BUILDER should be migrated so the next render is not the old rule |

## Code the docs-scripts inventory flagged that Docs does not own

* `rfx/studio/templates/patch_antenna.json` — draws one-cell ground and patch
  Boxes at `cell_size = 2 mm`. Its electrical cavity was 4 mm against a
  declared 2 mm substrate and halves to 2 mm under the contract, so the
  template's shipped resonance moves a lot. Studio itself has NO `two_plane`
  surface; its exposure is one level down.
* `rfx/experiments/canonical.py` — the `ExperimentSpec` geometry list accepts
  only `kind == "box"` and emits `sim.add(Box(...))`, so a Studio spec cannot
  express a sheet at all, and a physically honest 35 µm ground/patch thickness
  now RAISES. Someone must decide whether the spec gains a `thin_conductor`
  kind or Studio templates keep declaring plates.
* `rfx/pcb.py::Stackup` (public, in `rfx.__all__`) — the only production code
  path that emits sub-cell PEC Boxes by design. Under the §1.5 refusal the
  class is unusable for copper unless `to_shapes` emits sheets. Design note §6
  says `Stackup.to_shapes` puts a foil sheet on the dielectric INTERFACE it
  bounds, not on the foil's mid-plane; confirm that shipped.

## Frozen artifacts: label, do not edit

Committed run records quote pre-2.0 preflight text verbatim ("node-thin
conductor cell(s) carry statics that DISAGREE…", "one-plane", the `two_plane`
remedy). They are sha256-anchored evidence and editing one destroys what it is
for. `tests/contracts/test_two_plane_gone_from_docs_and_scripts.py` matches
them by path pattern (`_*_logs/`, `*.log`, `_NN_*_results/`) so no gate will
ever demand an edit to them:

* `scripts/diagnostics/_coax_msl_transition_settled_run_logs/*`
* `scripts/diagnostics/_msl_passive_port_reflection_logs/*`,
  `_mixed_refplane_logs/*`, `msl_z0_bias_floor_sweep/*`,
  `i517_mixed_solve_vs_ratio/*`, `i544_n_live_advisory_vs_assembler/*`,
  `msl_vi_flux_oracle/*`
* `docs/design_notes/patch_edgefed_s11_band_repin_{results,main,retired}.json`
  and their `.log` siblings
* `validation/crossval/_15_patch_results/*`, `_05_*` logs
* `tests/data/example_fidelity_snapshot.json` — NOT frozen; the examples group
  regenerates it last.
