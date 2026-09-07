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
* Producer script, on the branch and reviewable:
  `tests/unit/sparams/_results_931/capture_msl_sheet_thread_golden.py`. Two
  refusals, not warnings: it will not capture unless the trace is actually
  declared as a z-normal sheet realizing ONE wall plane, and it will not write
  unless the two captures are byte-equal.
* **VESSL run 369367259166 FAILED** (submitted 10:26 UTC, dead at 10:26:48
  before one line of the job ran):
  `/opt/vessl/scripts/<id>.sh: line 208: syntax error: unexpected end of file
  (expecting ")")`. Cause, verified: the first yaml carried the capture
  procedure as a `<<'PYEOF'` heredoc inside the `run:` block. The block itself
  is valid (`sh -n` on the extracted `run:` string exits 0), and none of the
  27 baseline yamls in `scratchpad/vessl_baseline/` uses a heredoc — VESSL's
  own run wrapper rewrites the block and the terminator stops terminating.
  Do not put a heredoc in a VESSL `run:` block; put the program in a file.
* **VESSL run 369367259224 FAILED** (14:24 UTC, rc=1 in 16 s). The shell was
  fine this time — the job ran, staged the checkout and reached the capture —
  and the capture died on `ModuleNotFoundError: No module named 'pytest'`:
  `test_msl_sheet_threading.py` imports pytest at module scope and the
  `rfx-openems` image does not carry it. The heredoc version would have hit the
  same wall had it started. Fixed by adding `pytest>=7.4` to the job's pip line;
  any future job that imports a test module needs the same.
* **VESSL run id: 369367259285** (resubmitted 2026-09-07 14:41 UTC,
  https://app.vessl.ai/remilab/runs/byungkwan/369367259285).
  Command: `vessl run create -f rfx-931-post-msl-sheet-golden.yaml` (the yaml
  sits beside this file; submit it from a non-git directory — the CLI cannot
  read a worktree's `.git` file). Run name `rfx-931-post-msl-sheet-golden`,
  preset `gpu-rtx4090`, `JAX_PLATFORMS=cpu`. Expected wall clock ~12 min
  (env install + two captures + a diff against the old golden).
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

## R8 — `test_lumped_twoport_vi_validation_battery.py` slow_physics gates

* Consumers: `test_thru_s11_floor`, `test_thru_s21_band_locks_shipped_decomposer_envelope`,
  `test_thru_s21_phase_band_is_sign_sensitive`, `test_thru_reciprocity`,
  `test_thru_passivity_singular_values` — all `@pytest.mark.slow_physics`,
  all fed by the module-scoped `thru_smatrix` fixture (one ~70 s solve).
* Why: the air-microstrip trace was a one-cell PEC Box, i.e. 0.5 mm of solid
  metal on a 1.0 mm ground-to-trace gap. It is a sheet now. The realized
  footprint does NOT move (17.0 x 5.0 mm, drawn, both before and after, and
  also under the pre-#931 rule). What moves is the port normalization: the
  port declares `extent = 1.0 mm`, exactly the gap, and `_wire_port_cells`
  rasterizes that endpoint-inclusive into THREE Ez edges, the third spanning
  1.0 -> 1.5 mm, above the trace. The volume trace shorted that surplus edge
  and the port counted 2 live cells; a foil does not, so `n_live` is 3 and
  `Z0_cell = Z0/n_live` goes Z0/2 -> Z0/3.
* **The gate BOUNDS are untouched.** What needs re-measuring is whether they
  still hold; if one does not, that is a result to report, not a bound to widen
  (the file's own tripwire protocol: "fail LOUDLY on the convention change,
  re-measure in the same PR").
* Command: `JAX_PLATFORMS=cpu python -m pytest
  tests/unit/sparams/test_lumped_twoport_vi_validation_battery.py -q -s
  -m slow_physics`
* Cost: cpu-min (~70 s thru + the DC anchor), 2 solves.
* **BLOCKED, deliberately.** The surplus live edge is a defect in the port's
  endpoint-inclusive extent rasterization that the volume trace was hiding. It
  belongs to the wire-port lane, and re-measuring the battery before it is
  settled would pin numbers taken with a port that is one edge too long. Order:
  settle `_wire_port_cells`' endpoint rule -> re-measure -> re-record the
  measured provenance in the module docstring.

## R9 — the openEMS referee's copy of the fixture's realized board

* Consumer: `tests/unit/sparams/test_probe_fed_msl_referee_contract.py::test_referee_record_still_describes_the_fixture_it_names`
  (build-time, `xfail(strict=True)`).
* Producer: `scripts/diagnostics/probe_fed_msl_openems_referee.py` —
  `RFX_REALIZED_RECORD` and `rfx_node_index`, both measured on the pre-#931
  `test_mixed_port_sparam` board (dx = 80 um, h_sub realized 320 um, trace
  480-560 um). That fixture is on-lattice now with a sheet foil: dx = h_sub/3,
  h_sub realized 254 um exactly, trace 508-592.67 um.
* **No solve.** Every number is a grid build, and no shipped test in the file
  depends on a physics value (the contract tests are arithmetic and structure).
  Replacement text with all of them:
  `docs/design_notes/931_migration/T2-probe_fed_msl_openems_referee.md`.
* Separately, and NOT a VESSL job: the referee's Stage-1 reproduce legs need
  re-running against the new board before it is next used as a comparator.
  openEMS is not installed on this pod; that is an openEMS run, not an rfx one.

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
