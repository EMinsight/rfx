# T2 (tests/unit/sparams, tests/unit/ports) — what must be re-solved for #931

Branch `feat/931-t2-sparams-ports`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-T2-sparams-ports`.

Every conductor in these two directories was re-declared under the lattice
ownership contract: foil is a **sheet** (a zero-thickness Box on the laminate
face), a plate/short/wall/pin is a **volume**, and boards that were drawn on a
mesh that bisects the laminate face are drawn **on-lattice** (`dx = h_sub/n`).
The build-time (no-solve) assertions are in the test files themselves and read
`realized_pec_edge_masks` through `tests/_realized_geometry.py`.

What follows is only the work that needs a **solve**. Nothing here is
hand-edited; each item names the producer that regenerates it.

## Machine rule

This pod is shared (load average 90-115 while this branch was written). Nothing
below runs here. One VESSL run per case, preset `gpu-rtx4090`,
`JAX_PLATFORMS=cpu` inside the job for CPU-produced fixtures, modelled on
`scratchpad/vessl_baseline/<case>.yaml`. The job reads the working tree as it
is, so the branch must be committed before submitting.

---

## R1 — `golden_msl_sheet_thread_{s,freqs}_13de212.npy` (SUBMITTED)

* Consumer: `tests/unit/sparams/test_msl_sheet_threading.py::test_o1_no_sheet_identity_vs_13de212_golden`
  (byte identity, `@pytest.mark.slow`).
* Why: the golden records a board this tree no longer builds. The trace was a
  one-cell PEC Box on a bisecting mesh (dx = 80 µm, h_sub/dx = 3.175, single
  wall at node 4); it is now a sheet on an on-lattice board (dx = h_sub/3, wall
  at node 3 = 254 µm). Both the realization and the mesh moved. Byte identity
  against the old file cannot hold and must not be relaxed into a tolerance.
* Producer: the module's own documented capture procedure —
  `build_msl_thru(sheet=None)` then
  `compute_msl_s_matrix(freqs=FREQS, num_periods=12.0)` on CPU float32 — run
  verbatim from the job and written to the fixtures directory.
* Cost: ~3 min CPU per capture, captured twice to confirm determinism.
* Outputs: `tests/fixtures/golden_msl_sheet_thread_s_931.npy` and
  `..._freqs_931.npy`, plus the two pre-#931 files kept beside them as history.
* **VESSL run id: 369367259166** (submitted 2026-09-07 10:26 UTC,
  https://app.vessl.ai/remilab/runs/byungkwan/369367259166).
  Command: `vessl run create -f rfx-931-post-msl-sheet-golden.yaml` (the yaml
  sits beside this file; submit it from a non-git directory — the CLI cannot
  read a worktree's `.git` file). Run name `rfx-931-post-msl-sheet-golden`,
  preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu`. Expected wall clock ~12 min
  (env install + two captures + a diff against the old golden). It asserts
  the trace really is declared as a sheet, and refuses to write a golden if
  the two captures are not byte-equal.
  Artifacts: `/root/workspace/claude-workspace/rfx/runs/issue931-post-msl-sheet-golden-<ts>/`.
* Ingest: copy the two `.npy` files onto the branch, point the test's
  `_FIXTURES` load at them, and quote the max deviation against the pre-#931
  golden in the commit (it will be large — the board changed).

## R2 — `tests/unit/sparams/test_msl_port_integration.py` gate values

* Three `@pytest.mark.slow` gates: `mean|S11| < 0.15`, `mean|S21| ∈ (0.90, 1.05)`,
  `mean Re(Z0) ∈ (40, 65) Ω`. **The bounds are untouched.** Only the recorded
  MEASURED values (0.118 / 0.972 / 54 Ω, refreshed to 0.1160 / 0.9930 / 57.58 Ω)
  are pre-contract and must be re-measured.
* Why: the fixture moved from dx = 80 µm (the strip at z = 320 µm over a 254 µm
  dielectric plus a 66 µm air gap) to dx = h_sub/3 with the trace declared as a
  sheet on the laminate face. The file's own docstring already records the
  aligned-mesh sibling at **44.11 Ω**, which is inside (40, 65) — that is why
  the windows are expected to hold without moving. If one does not hold, that
  is a result to report, not a bound to widen.
* Command: `JAX_PLATFORMS=cpu python -m pytest
  tests/unit/sparams/test_msl_port_integration.py -m slow -q -s`
* Cost: cpu-hour class (three thru solves plus the length-invariance ladder).
* Not yet submitted — submit after the branch is merged with the preflight
  group's work, so the run measures one geometry and one advisory set.

## R3 — `tests/unit/ports/test_msl_source_fixture_static.py` highmem referee

* The `@pytest.mark.highmem` AD/FD referee gate (0.03 on the f32 mini-referee)
  was measured on the bisecting board; `_DX` is now `h_sub/3`.
* Command: `JAX_PLATFORMS=cpu python -m pytest
  tests/unit/ports/test_msl_source_fixture_static.py -q -m highmem`
* Cost: cpu-hour, 3-24 GB RSS — the weekly highmem lane, not this pod.

## R4 — `tests/fixtures/msl_replay_*` — BLOCKED on a script this group does not own

* Producer: `scripts/capture_msl_replay_fixture.py` (line 71 still draws the
  trace as a one-cell Box `H_SUB -> H_SUB + DX` at dx = 80 µm).
* Consumer: `tests/unit/autodiff/test_msl_sparam_ad.py` (group T1's directory).
* Capturing now would freeze a VOLUME trace on a bisecting mesh — the geometry
  this branch is removing. The replacement text for the script is in
  `docs/design_notes/931_migration/T2-capture_msl_replay_fixture.md`; the
  capture is a ~10 min CPU job once that lands.

## R5 — `tests/fixtures/thru_singular_value_dx_ladder/rung_dx_over_{1,2,4}.json`
     — BLOCKED on a script this group does not own

* Producer: `scripts/diagnostics/thru_singular_value_dx_ladder.py`
  (`--dx-divisor {1,2,4} --output ...`), whose trace is still a one-cell Box
  (`H_M -> H_M + dx`). The trace height is 1.0 mm and dx is 0.5/0.25/0.125 mm,
  so the board is already on-lattice; only the declaration changes.
* Consumer: `tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py`
  (this group). Its G4 gate reads `finite_pec_cells == [340, 1360, 5440]` and
  `live_flags[-1] is False`; under a sheet declaration a conductor owns no cell,
  so the first collapses to zero and the second becomes True. G4 is restated on
  the realized sheet FOOTPRINT (which scales dx⁻² the same way) — see the test's
  own docstring — and cannot be re-pinned until the rungs are re-captured.
* The three rung JSONs are frozen VESSL run records (run 369367257803): they get
  a NEW record, not a value edit, exactly as the coax predeclaration does.
* Replacement text for the script:
  `docs/design_notes/931_migration/T2-thru_singular_value_dx_ladder.md`.
* Cost when unblocked: three runs, ~gpu-hour class at divisor 4.

## R6 — `tests/unit/sparams/test_coax_msl_transition.py` — a NEW attempt record

* The junction's realization changes on every axis of it: the ground plane and
  the trace were `_half_cell_box_z(n, n)` recipes emulating a sheet with an
  integer plane index (they are sheet declarations now), the pin Cylinder gains
  its far end plane, and `_TRACE_Y_LO/HI_OFFSET_NODES = -3/+2` is a hi-face
  compensation of the class the contract forbids.
* The PREDECLARATION / PREDECLARATION_ATTEMPT2 blocks and
  `SETTLED_RUN_RECORD` (VESSL 369367252283, settling_db [-45.94, -44.17]) are
  frozen and **must not be edited**. The post-contract junction needs its own
  attempt predeclaration, then a GPU run, the way the RASTERIZER NOTE already
  marks the pre-#834 records.
* Not attempted on this branch: it is a full re-declaration of a 3-D conductor
  stack plus a new predeclaration, and it is the single most expensive fixture
  in the group. Listed, not run — see the report.

## R7 — `msl_z0_bias_floor_sweep_realized_anchor.json` — cross-group, blocked

* `tests/unit/sparams/test_msl_z0_bias_floor_sweep_realized_anchor.py` re-derives
  `_POST_802_W_UM` live from `fidelity_report()`, and
  `tests/unit/ports/test_msl_port_preflight.py::test_realized_thickness_advisory_threshold_is_derived_not_invented`
  derives check 2c's SENS/TOL from the same artifact. The artifact is produced by
  `scripts/diagnostics/msl_z0_bias_floor_sweep_realized_anchor.py` from the
  sha256-frozen `msl_z0_bias_floor_sweep.json`, which must not be edited.
* Order (unchanged from the inventory): settle what `realized_extent_um` means
  for a sheet vs a volume in `fidelity.py` → re-solve the 6-point sweep →
  regenerate the anchor → re-derive SENS/TOL. Steps 1 and 2 are outside this
  group.

---

## Note on `dx = h_sub/3` vs `h_sub/4`

Preflight's substrate-resolution check wants `dx ≤ 63.5 µm` (4+ substrate
cells) AND `h_sub/dx` integral. `h_sub/3 = 84.67 µm` satisfies the second and
not the first, so the aligned fixtures still carry that advisory. It is an
ACCURACY advisory and orthogonal to the ownership contract: `h_sub/4` would
silence it at ~2.4x the cell count and ~1.3x the step count on every solve in
the group. The cheaper aligned mesh is used deliberately; a fixture whose gate
turns out to need four cells should move to `h_sub/4` on its own evidence, not
to quieten a warning.
