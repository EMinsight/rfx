# T3 — tests/unit/{nonuniform,runners,grid,subgrid} under the lattice ownership contract

Branch `feat/931-t3-nu-runners-grid-subgrid`. Inventory: `tests-nu-runners-grid-subgrid.json`
(47 sites), corrected by `critic.json`. This note records the decisions the design
note left to the migrating agent, and the sites that belong to another owner.

## Decisions taken here

The inventory asked the PI four questions. Three are answered by the design
note as it stands; the fourth is answered by the branch. All four are written
into the fixtures themselves so the next reader does not have to re-derive them.

1. **Does the ownership change reach MATERIAL Box sampling?** No. §1.8:
   "Dielectric sampling (node, half-open) … unchanged". So
   `test_vmap_sweep_dft_planes.py::test_exact_hi_face_touch_matches_run_via_shared_fallback`
   keeps its hi-face-drop assertion, and the #627 CPML pad fallback and #655
   node repair stay live. The fixture says it is fenced, so the assert is not
   mistaken for an oversight. rfx does carry two conventions for one `Box`
   primitive — centre-sampled for a PEC volume, node half-open for a
   dielectric — and §1.1 states that on purpose (they coincide for
   node-aligned corners).

2. **Is a high-sigma MATERIAL box a conductor under the contract?** Two
   different answers, and the difference is the API level:
   * `sim.add(shape, material=<sigma above threshold>)` goes through
     `classify_pec_entry` and IS a declared conductor — that is the WR-90
     iris in `test_nonuniform_pec_scatterer_limit.py` (`sigma=1e7`);
   * a raw `sigma` stamp into the material arrays — `box.mask(grid)` +
     `jnp.where(mask, 1e10, sigma)` in `test_simulation.py`'s T-junction, and
     `rasterize(grid, [(plate, 1.0, PEC_SIGMA)])` in
     `test_x64_scan_carry_dtypes.py` — is a sigma FILL, fenced out by §1.8 as
     a lossy volume model.
   Both files now say which they are. The T-junction's declared `a = 0.04`
   against a realized 0.042 channel is the #868 class and stays open under
   the follow-up issue §1.8 names; it is not repaired here and not papered
   over.

3. **Does (S) cover LOSSY thin conductors?** No. §1.8 fences the DC fold
   (`sigma_bulk < 1e6`, no `f0`) out as a lossy volume model. So
   `test_vmap_sweep.py::_make_cpml_sim`'s conductor keeps its finite 2x2-cell
   box, `test_thin_conductor_fixture_is_live`'s `sigma_eff = 175 S/m`
   assertion stands, and the calibrated `n_steps = 200` gate needs no
   re-measurement. The word "sheet" is removed from those docstrings: the box
   is two cells thick in BOTH y and z, so it has no single normal axis and
   was never a sheet — only the pre-#931 rule made it look like one.

4. **Are DOMAIN FACES inside the realized-edge-set function?** No, §1.8.
   `test_adi.py::test_adi_cavity_resonance`'s `a_eff = (Nx-1)*dx` and
   `test_distributed.py::TestLegacyPmapPadXFaceAppliers` are domain-boundary
   PEC and unchanged. Both now say so, because a domain wall and a body wall
   really are two mechanisms and the compensation in the ADI oracle looks
   exactly like the body-rule compensations the contract abolishes.

Two more the design note did not raise:

5. **`conformal_pec` is not a realization knob.** §1.5 forbids per-ENTRY
   realization keywords; `conformal_pec` is a run-level subpixel model
   (Dey-Mittra) layered on the realized edge set and fenced by §1.8, so it is
   not in the `two_plane` deletion list. It is still honoured on the uniform
   lane and warn-dropped on NU/subgrid/distributed, i.e. the lanes agree
   about realization and disagree about subpixel treatment.
   `test_silent_drop_warnings.py` records the distinction.

6. **`pec_mask_override` is a VOLUME override.** So the geom|override union
   in `test_distributed_nu_kernel.py` is a union of cell occupancies and
   stays a boolean OR. A sheet cannot be expressed through it; the
   distributed-NU forward lane refuses one loudly instead
   (`test_distributed_nu_pec_mask_lane_parity.py`).

## Sites that are another owner's

* `rfx/api/_preflight.py` — the "declared conductor realizes no wall plane"
  ERROR (inventory MISSING GATE 1) is preflight's, owner **P**. The test-side
  half is done: the two fixtures in this group that declared a conductor and
  realized NOTHING (`test_nu_progress_chunking.py`, the two zero-thickness
  thin conductors in `test_auto_config.py`) now assert a realized wall plane
  at build time, so they can no longer pass vacuous.
* `test_inplane_grading_guards.py`'s under-resolution advisory sentence ("a
  1.5 mm PEC volume in a 500 um-cell region is 3 cells across") is scored by
  preflight from the body. The fixture is redrawn on nodes and asserts the
  realized span; whether preflight scores it from realized walls or from the
  Box mask is owner **P**'s call. The assertion here is qualitative and fires
  either way.

  **Finding for owner P, measured while redrawing it.**
  `_validate_mesh_quality`'s `_local_cell` resolves a body whose face lands
  exactly on a band boundary to the FINER neighbouring band. On the fixture's
  profile (`[12x250um, 8x500um, 8x125um, 12x250um]`, coarse band
  `[3.0, 7.0) mm`) the same 3-coarse-cell PEC volume scores:

      drawn 3.0 -> 4.5 mm (lo face ON the band's first node): 0 advisories
      drawn 3.5 -> 5.0 mm (one coarse cell inside):           2 advisories

  Same body, same 3 cells, same realized-vs-drawn span. The 3.0 mm case is
  scored against the 250 um cell on the fine side of the boundary — the
  exact "a body in a coarse region judged by a fine cell it never sees"
  failure the #743 check exists to prevent, reappearing at the band edge.
  The fixture is drawn one coarse cell inside the band so it keeps testing
  what it names; the tie itself is NOT worked around and is not fixed here.
* `validation/crossval/05_patch_antenna.py` and
  `scripts/diagnostics/patch_tutorial_rfx.py` — owner **X-A**.
  `test_patch_uniform_fine_substrate.py::build_uniform_fine_z` is the
  grid-build lock those two imitate, and it has dropped its two reserved
  metal cells here (see below). The live cv05 migration is X-A's.
* `tests/unit/sparams/test_settling_witness.py` keeps its own copy of the
  `_msl_thru` geometry migrated in `test_run_progress_reporting.py`; owner
  **T (sparams group)**.
* `rfx/surrogate.py::export_geometry_sdf` iterates `sim._geometry` only, so a
  sheet declared through `add_thin_conductor` never reaches it and vanishes
  from exported training data with no error. A zero-thickness Box DOES
  register (the exporter's containment test is closed), which is what
  `test_amr_surrogate.py` now pins. **The `add_thin_conductor` gap is a real
  defect and is not fixed here — `rfx/surrogate.py` is nobody's file in this
  phase.**

## Recompute

See `RECOMPUTE.md` beside this file.
