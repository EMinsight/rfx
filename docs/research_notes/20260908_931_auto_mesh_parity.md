# #931 auto-mesh parity review and verification

Worktree: `/root/workspace/byungkwan-workspace/research/rfx-931-fastlane-reds`.
Branch: `feat/931-automesh-parity`; starting commit `af164f8079b1fd5adf0dc785f2ea90af87bc6fe3`.
All implementation, scripts, tests and artifacts stayed in this worktree.
Scripts used `PYTHONPATH=$PWD`; import inspection confirmed this worktree's
`rfx/__init__.py`. All JAX work used `JAX_PLATFORMS=cpu`. Only one pytest
invocation ran at a time, with at most four workers and `timeout 1200`.

## Audit before edits

At the starting commit, `_auto_configure_mesh()` had one caller, inside `run()`.
The following consumers could see unresolved inputs on a fresh `dx=None`
simulation with geometry. The inventory was reported before editing.

| Surface | Implementation | Original discrepancy |
|---|---|---|
| `run` | `api/_execute.py` | Already resolved before dispatch; reference behavior |
| `forward` | `api/_execute.py` | Profile-based dispatch and grid assembly used unresolved inputs |
| `optimize`, `progressive_optimize`, `gradient_check` | `optimize.py` | Region sizing and forward baseline could disagree with run; progressive stages with explicit dx already avoid this |
| `topology_optimize` | `topology.py` | Unconditional uniform grid and kernel |
| `preflight`, `preflight_sparameters` | `api/_preflight.py` | Config, geometry, resource and lane checks preceded resolution |
| `fidelity_report` | `fidelity.py` | Declared-versus-realized audit selected the default grid |
| `conductor_mask`, `validate_subgrid` | `api/_compile.py`, `api/__init__.py` | Inspected unresolved geometry; subgrid validation is uniform-only |
| `compute_waveguide_s_matrix`, `compute_msl_s_matrix` | `api/_sparams.py` | Lane selection and port/reference placement preceded resolution |
| `compute_mixed_s_matrix`, `compute_coaxial_s_matrix`, `compute_coaxial_line_reflection`, `compute_coaxial_two_port`, `compute_coax_msl_transition` | `api/_sparams.py` | Uniform-lane eligibility could be decided before an automatic profile existed |
| `compute_lumped_wire_s_matrix_via_scan` | `probes/sparam_driver.py` | Unconditional uniform grid and scan |
| MSL wave/plane probe registration | `probes/msl_wave_decomp.py` | Spacing read or uniform grid before resolution |
| `vmap_material_sweep` | `vmap_sweep.py` | Uniform assembly happened before fallback eligibility |
| AD memory estimate, explanation, plan, preflight and compiled-certificate preflight | `api/__init__.py` | Cell counts, profiles and lane sizing used unresolved inputs |
| `mesh_intelligence_report`, `plan_mesh`, `repr(sim)` | `api/__init__.py` | Raw mesh reads; planner timestep/repr unconditionally built uniform grids |
| `profile_forward` | `profiling.py` | Preliminary grid selection differed from execution; also ignored x/y-only profiles |
| Geometry slice, rasterized slice, stack profile, structure/domain and far-field views | `visualize.py` | Unresolved profile selection; geometry slice was unconditionally uniform |
| Field screenshots | `visualize3d.py` | Uniform grid and scalar coordinate spacing |
| Dashboard previews and field views | `dashboard/app.py` | Uniform preview/plot grid |
| Scene artifacts, runtime reports and bundles | `artifacts.py`, `api/_artifacts.py` | Mesh metadata described unresolved inputs |
| Geometry JSON, experiment reports, simulation datasets | `io.py` | Unresolved mesh metadata |
| Design IR export, including resolved MSL offsets | `interop/_design.py` | Export before run differed from export after run |
| Source, MSL/waveguide/Floquet port, refinement and monitor registration | `api/__init__.py` | Construction-time spacing, lane and domain decisions could precede run |
| Material-fit template | `differentiable_material_fit.py` | Preliminary grid build |
| `auto_refine` fallback | `amr.py` | Uniform fallback when no result grid was available |
| RIS substrate construction | `ris.py` | Spacing estimate read before run |

`parametric_sweep` and `convergence_study` delegate execution to `run` and
already benefit there. `quick_convergence` explicitly supplies candidate dx;
its preliminary source-domain/profile reads still need the resolved view.
Low-level functions handed an already constructed Grid are outside the
Simulation auto-configuration boundary.

## Shared resolution and decisions

`api/_mesh.py::_resolve_mesh()` owns resolution. Mesh descriptors route reads
of `_dx`, `_domain` and all three profiles through it, including reads made
before a caller chooses its solver lane. Their dictionary slots retain the
declared values. This covers new callers without a maintained entry-point list.
`_build_realized_grid()` selects the actual lane; `_build_grid()` is explicitly
uniform-only and refuses a profiled mesh, preventing new uniform consumers
from silently constructing a surrogate.

* **Idempotency:** a cache holds strong references to immutable geometry and
  material records plus mesh/frequency/boundary inputs. Unchanged declarations
  reuse the resolution. Builder additions and record replacements invalidate
  it, including after preflight or a construction-time preview. Tests compare
  preflight-then-run with run alone and verify identical mesh/timestep.
* **Differentiation:** auto-configuration runs under
  `jax.ensure_compile_time_eval()` from static inputs. Tests first enter
  `forward` inside `jit(grad(...))` on both uniform and automatically graded
  models, and require finite nonzero derivatives and concrete cached values.
  Traced geometry/materials require explicit mesh inputs. Actual traced
  profiles keep their differentiable build path.
* **Preflight side effects:** declared dx/domain/profiles are unchanged.
  Preflight does populate a derived cache and can emit a selection warning;
  it is not claimed to be observationally pure. This avoids freezing partially
  constructed geometry while making its legality decision agree with execution.
* **Warnings:** `AutoMeshWarning` subclasses `UserWarning`. Selection and
  auto-configuration notices are emitted on first resolution, outside
  preflight's captured legality findings. Mesh-quality validators continue to
  produce normal findings. Selection information is not an uncoded error or
  an order-dependent preflight issue.
* **Unsupported lanes:** topology optimization, subgrid validation/refinement
  and the low-level uniform lumped/wire scan refuse a resolved NU mesh. Public
  S-parameter lanes retain their existing support fences, now evaluated after
  resolution. Material sweeps take their existing sequential NU fallback.
  This does not add numerical support to a uniform-only kernel.

The NU tracing test found an additional traced-boolean conversion in
`rasterize_geometry`. Its optional PEC-mask decision now uses the same static
fallback as the uniform assembler. Static NU grids also build outside tracing
so conductor classification retains concrete coordinates. A regression requires
an outer JIT to refuse an actual sub-cell PEC volume on a static NU grid.
No tolerance, lattice ownership rule or sub-cell refusal was weakened.
A final outer-JIT check with preflight enabled exposed a static Box mask
converted to numpy inside the graded-geometry validator. Preflight now runs
its static diagnostics under the same host-evaluation context; regression
cases cover both preflight enabled and disabled on both grid lanes.

The uniform-builder fence exposed a pre-existing uniform approximation in
`_wire_port_cell_centers`. It now uses the NU runner's endpoint snap, shared
edge span and physical node coordinates. NU dead-cell classification remains
unsupported and continues to emit its explicit unavailable advisory; the
correction does not claim to implement that classifier.

## Documentation witnesses and limits

The original materials-geometry plate declaration (base revision line 52)
resolves to dx = 0.5 mm, 172 physical z cells, padded shape `(133, 133, 205)`,
and dt = `4.119817003110024e-13` s. Preflight emits no PEC legality error.
Both public execution entries complete an actual one-step solve on fresh
instances. These are legality/runtime checks, not converged RF measurements;
the documentation block has no excitation.

The chained-material block (base line 74) and copper-via block (base line 200)
are appended verbatim to that base declaration in the regression fixtures.
Both pass preflight and real material assembly through `run` and `forward`.
Those larger variants stop immediately after assembly; their scans were not
run. Explicit-dx controls require `pec_box_subcell` at error severity and
require both entries to raise, making “run()/forward() will raise” true at the
refusal boundary.

The first synthetic AD fixture used a 1 A source: primal loss was finite
(`6.960453e17`) but the float32 reverse pass returned NaN. At 1 microamp the
same resolved model gave loss `696045.3` and derivative `-185334.86`; a unit
field source also gave finite loss/derivative (`15.011611`, `3.5086882`). The
regression uses microamps to avoid this unrelated dynamic-range artifact.

No GPU, distributed solve, converged S-parameter campaign, or external CAD/
PyVista/VTK backend was validated. Existing viewers accepting an explicit Grid
do not participate in automatic Simulation resolution. The declaration cache
assumes frozen geometry/material records are not mutated through nested arrays.

## Verification ledger

Final status: **1,063 passed, 0 failed, 3 skipped, 1 expected failure** across
unique selected tests, using the latest result for each test. This is a union
of the focused, regression and final rerun results, not a claim that the full
repository suite ran. After the default-preflight tracing addition, all 77
selected tracing/preflight cases passed. The broad invocation reported 852 passed / 3 failed /
1 skipped / 1 xfailed; the three failures were corrected and the final
invocation passed all 211 selected cases. The focused cases outside that
invocation passed on their latest run.

The three broad failures were: fallback warning capitalization (existing text
preserved); a test deliberately constructing the former uniform surrogate
(now explicit via Grid); and a planner fixture with invalid boundary profile
endpoints (fixed, with an additional rejection test). Two initial visualization
fixture errors and two new wire-coordinate fixture errors were corrected.
The initial AD source dynamic-range issue and the NU traced-boolean defect are
described above. No failed gate was accepted or tolerance relaxed.

Snapshot regeneration processed all **51 variants** in 68.7 seconds and produced
**byte-identical** `tests/data/example_fidelity_snapshot.json`. The snapshot
contract file passed all 176 tests. An intermediate regeneration had removed
five NU wire-classifier unavailable advisories; that was investigated and fixed
in the coordinate helper, not accepted as a new baseline.

All new Python files pass Ruff. Checking every changed Python file still reports
six pre-existing findings (verified against the starting revision): one E741 in
`_execute.py`, one F401 in `_design.py`, two E731 in `visualize.py`, and two F401
in `test_auto_config.py`. `git diff --check` passes.

The default repository marker filter excludes GPU/slow/slow_physics tests.
`test_optimize.py`, `test_topology.py`, and `test_vmap_sweep.py` were requested
in the broad invocation but are GPU-marked and contributed no executed tests.
CPU optimization, gradient check, sweep fallback, and the topology NU refusal
are instead exercised by the new consumer tests and the other selected files.
Plotly accounts for all three skips (two tests plus one module collection).
The existing strict xfail concerns the wording of the wire-port end-gap
advisory in `test_mixed_port_sparam.py`; its physical premise passes.

A separate pytest started elsewhere on the shared pod during the broad run.
It was left untouched; no next pytest here started until both invocations had
finished. Test durations include multiple solves: the longest paired-driver
test took 143.1 seconds; new one-step documentation solves stayed well below
one minute each.

### Exact results by test file

Counts are whole-file selected outcomes; class-parametrized cases are summed.
No row below represents source-line coverage.

| Test file | Passed | Failed | Skipped | Xfailed |
|---|---:|---:|---:|---:|
| `tests/contracts/test_example_fidelity_contract.py` | 176 | 0 | 0 | 0 |
| `tests/contracts/test_lattice_ownership_contract.py` | 114 | 0 | 0 | 0 |
| `tests/studio/test_interop_design_document.py` | 200 | 0 | 0 | 0 |
| `tests/studio/test_interop_design_schema_contract.py` | 29 | 0 | 0 | 0 |
| `tests/unit/api/test_artifacts.py` | 13 | 0 | 0 | 0 |
| `tests/unit/api/test_dashboard.py` | 7 | 0 | 0 | 0 |
| `tests/unit/api/test_rasterized_slice_viewer.py` | 29 | 0 | 0 | 0 |
| `tests/unit/api/test_visualize.py` | 4 | 0 | 0 | 0 |
| `tests/unit/api/test_visualize3d.py` | 6 | 0 | 0 | 0 |
| `tests/unit/api/test_visualize_3d.py` | 0 | 0 | 1 | 0 |
| `tests/unit/api/test_visualize_realized_grid.py` | 4 | 0 | 2 | 0 |
| `tests/unit/autodiff/test_ad_memory_grid_and_sheet.py` | 15 | 0 | 0 | 0 |
| `tests/unit/autodiff/test_estimate_ad_memory.py` | 73 | 0 | 0 | 0 |
| `tests/unit/autodiff/test_nonuniform_forward_grad.py` | 11 | 0 | 0 | 0 |
| `tests/unit/autodiff/test_nonuniform_gradient.py` | 4 | 0 | 0 | 0 |
| `tests/unit/autodiff/test_optimize_nonuniform.py` | 1 | 0 | 0 | 0 |
| `tests/unit/geometry/test_fidelity_report.py` | 26 | 0 | 0 | 0 |
| `tests/unit/geometry/test_rasterization_coordinate_exactness.py` | 65 | 0 | 0 | 0 |
| `tests/unit/grid/test_auto_config.py` | 27 | 0 | 0 | 0 |
| `tests/unit/grid/test_auto_mesh_consumers.py` | 6 | 0 | 0 | 0 |
| `tests/unit/grid/test_auto_mesh_resolution.py` | 18 | 0 | 0 | 0 |
| `tests/unit/nonuniform/test_dz_only_dispatch_contract.py` | 12 | 0 | 0 | 0 |
| `tests/unit/nonuniform/test_mesh_intelligence_report.py` | 6 | 0 | 0 | 0 |
| `tests/unit/nonuniform/test_mesh_planner.py` | 9 | 0 | 0 | 0 |
| `tests/unit/nonuniform/test_nu_wire_port_lane_parity.py` | 23 | 0 | 0 | 0 |
| `tests/unit/preflight/test_auto_preflight.py` | 3 | 0 | 0 | 0 |
| `tests/unit/preflight/test_nonuniform_wire_port_centers.py` | 3 | 0 | 0 | 0 |
| `tests/unit/preflight/test_preflight_rasterization.py` | 44 | 0 | 0 | 0 |
| `tests/unit/preflight/test_preflight_reads_realized_edges.py` | 6 | 0 | 0 | 0 |
| `tests/unit/preflight/test_run_preflight_parity.py` | 3 | 0 | 0 | 0 |
| `tests/unit/runners/test_profiling.py` | 2 | 0 | 0 | 0 |
| `tests/unit/runners/test_vmap_sweep_dft_planes.py` | 61 | 0 | 0 | 0 |
| `tests/unit/runners/test_vmap_sweep_eligibility.py` | 5 | 0 | 0 | 0 |
| `tests/unit/sparams/test_auto_mesh_driver_purity.py` | 3 | 0 | 0 | 0 |
| `tests/unit/sparams/test_mixed_port_sparam.py` | 29 | 0 | 0 | 1 |
| `tests/unit/sparams/test_sparam_driver_matches_eager.py` | 5 | 0 | 0 | 0 |
| `tests/unit/sparams/test_waveguide_nu_sparam.py` | 8 | 0 | 0 | 0 |
| `tests/unit/subgrid/test_amr_surrogate.py` | 13 | 0 | 0 | 0 |

### Coverage of every changed implementation/documentation file

The exact pass/fail counts for the named test files are in the table above.
Production and prose files do not themselves have pytest pass/fail counts.

| Changed file | Verification |
|---|---|
| `rfx/api/_mesh.py` | auto_mesh_resolution, auto_mesh_consumers, auto_config; interop round trips |
| `rfx/api/_compile.py` | lattice_ownership_contract, auto_mesh_resolution/consumers, rasterization_coordinate_exactness |
| `rfx/api/_execute.py` | auto_mesh_resolution (both real documentation solves), run_preflight_parity, auto_preflight |
| `rfx/api/__init__.py` | auto_mesh_consumers/driver_purity; mesh_planner, memory and interop tests |
| `rfx/api/_preflight.py` | nonuniform_wire_port_centers, preflight_rasterization, preflight_reads_realized_edges, mixed_port_sparam |
| `rfx/api/_sparams.py` | auto_mesh_driver_purity, mixed_port_sparam, waveguide_nu_sparam |
| `rfx/geometry/rasterize_grid.py` | auto_mesh_resolution JIT/grad/refusal; lattice_ownership_contract, rasterization_coordinate_exactness |
| `rfx/runners/nonuniform.py` | auto_mesh_resolution; nonuniform_forward_grad, nonuniform_gradient, nu_wire_port_lane_parity |
| `rfx/optimize.py` | auto_mesh_consumers real gradient_check and optimize; optimize_nonuniform |
| `rfx/topology.py` | auto_mesh_consumers NU refusal; lattice_ownership_contract CPU topology witness |
| `rfx/vmap_sweep.py` | auto_mesh_consumers; vmap_sweep_eligibility, vmap_sweep_dft_planes, lattice_ownership_contract |
| `rfx/probes/sparam_driver.py` | sparam_driver_matches_eager |
| `rfx/profiling.py` | profiling |
| `rfx/amr.py` | amr_surrogate; shared uniform-builder refusal test |
| `rfx/dashboard/app.py` | dashboard; realized-grid/timestep consumer tests; no live browser session |
| `rfx/interop/_design.py` | interop_design_document, interop_design_schema_contract, auto_mesh_consumers |
| `rfx/visualize.py` | visualize_realized_grid, visualize, rasterized_slice_viewer, lattice_ownership_contract |
| `rfx/visualize3d.py` | visualize_realized_grid, visualize3d; Plotly module unavailable |
| `docs/public/guide/materials-geometry.mdx` | literal base/chained/via declarations in auto_mesh_resolution; prose reviewed |
| `docs/design_notes/20260906_plan_realign_lattice_ownership.md` | amendment checked against implementation and lattice_ownership_contract |
| This report | counts extracted from pytest JUnit XML; snapshot diff and source/test mapping reviewed |

The changed test files themselves are the eight corresponding rows above:
`test_auto_config`, `test_auto_mesh_resolution`, `test_auto_mesh_consumers`,
`test_visualize_realized_grid`, `test_auto_mesh_driver_purity`,
`test_nonuniform_wire_port_centers`, `test_ad_memory_grid_and_sheet`, and
`test_mesh_planner`: **85 passed, 0 failed, 2 skipped** in total.

## Pre-commit audit

The repository memory files required by ancestor guidance (`index.md`,
`rfx-known-issues.md`, `rf_capability_audit_2026-05-03.md`) are absent in this
worktree; no other worktree was inspected to fill that gap. No conflicting
entry was found. The existing design note's §1.7 "One source, every consumer"
is the local normative constraint, and this resolution boundary follows it.
R2-attempts = 0 (architectural correction, not another physics-parameter
attempt). Implementation/test failures above were diagnosed with concrete
falsifiers before fixes. The cheap falsifier exercised is the documented PEC
plate's run/forward/preflight agreement, with actual sub-cell refusal controls
and the §5 lattice contract tests. No remaining merge blocker was identified.
