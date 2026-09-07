"""cv15's mandatory geometry self-check: the REALIZED electric-wall PLANES
must be exactly the declared ones, measured through the contract's one edge
set.

History. Issue #740: cv15's ``#325 AVOIDANCE (mandatory)`` self-check asserted
only the substrate's z EXTENT (``n_sub_raster == N_SUB``, from ``round(z/DX)``
cell counting) and kept PASSING while the ground conductor realized its
electric wall one full cell BELOW the declared substrate floor (the #693
"vacuum ground cell" trap) -- a live vacuum cell inside the modelled cavity,
undetected, +55.0% of the cavity's electrical thickness.

Issue #931 (the lattice ownership contract): the ground and the patch are FOIL
and are now DECLARED as sheets -- zero-thickness PEC ``Box``es on the two
substrate faces, the same structure both openEMS legs build. The check they
are under is no longer a hand-copied edge rule; ``assert_realized_stack`` reads
``rfx.boundaries.pec.realized_pec_edge_masks`` through ``realized_wall_planes``
and asserts three things: a wall at each declared plane, NO wall anywhere else
(in particular none at ``k_patch + 1``), and that the sheets wrote no material.

The geometry under test is built through cv15's OWN production builder,
``build_rfx_sim(*, do_gain, ground_plane_z, patch_kind)`` -- separated from
``run_rfx()`` for the #740 review (cv15 is classified ``audited`` in
``tests/_example_fidelity_lib.py`` on that builder). The positive tests pass NO
declaration arguments, so the script's own declarations are what is under test.
The first version of this file mirrored the geometry in a test-local copy that
hardcoded the fix, and deleting the fix from the script left it green -- the
reviewer's finding, and why the builder exists. ``assert_realized_stack`` and
``_stack_check_ok`` are called UNMODIFIED from the script itself, never copied.

Both negative controls run through that same production builder:

* ``ground_plane_z = z_sub_lo - DX`` -- the ground sheet declared one node
  plane low. It reproduces the pre-#740 realization (one wall below the floor,
  a live vacuum cell in the cavity) and keeps the geometry behind
  ``_15_patch_results/rfx_one_plane_ground_b29f9de7.json`` -- the +6.09%
  blindness evidence two public documents cite -- reachable through the public
  API after ``two_plane`` is deleted.
* ``patch_kind = "volume_1cell"`` -- the pre-#931 patch spelling, a one-cell
  PEC ``Box``. Under the contract a one-cell Box is a filled slab with BOTH
  faces, so it grows a wall at ``k_patch + 1`` (11.9062 mm) that the openEMS
  zero-thickness patch has no counterpart for. Without this arm the
  no-extra-wall assertion is decoration: nothing reachable would make it fire.

cv15 is guarded by ``if __name__ == "__main__":`` (see its final lines), so
importing it (to reach its module constants and the two functions above)
executes no simulation. Every test here is build-time -- ``_build_grid`` +
``_assemble_materials`` + the edge masks. No solve.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"


def _load_cv15():
    """Import cv15 as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("_cv15_wall_planes", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_test_sim(cv15, **kw):
    """Build cv15's geometry through the PRODUCTION builder,
    ``cv15.build_rfx_sim`` -- not a test-local mirror (#740 review, item 1).

    No keyword (what every positive test passes) means the script's own
    declarations are under test: change either conductor's declaration in the
    script and the positive tests below go red. Only the negative controls
    pass a keyword. Returns ``(sim, grid, patch_shape)`` so the caller can run
    the script's OWN ``assert_realized_stack`` against it, cheaply (no solve).
    """
    sim, patch_shape, _geom = cv15.build_rfx_sim(do_gain=False, **kw)
    grid = sim._build_grid()
    return sim, grid, patch_shape


# ---------------------------------------------------------------------------
# assert_realized_stack: on the real rasterized geometry, via the contract's
# one edge set (cheap -- _build_grid + _assemble_materials, no solve).
# ---------------------------------------------------------------------------

def test_cv15_committed_geometry_realizes_declared_walls(capsys):
    """The committed geometry must realize its electric walls exactly at the
    declared z_sub_lo/z_sub_hi planes, across the whole patch footprint."""
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(cv15)
    stack_check = cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()

    assert stack_check["ground_wall_z"] == pytest.approx(cv15.AIR_BELOW, abs=1e-12)
    assert stack_check["patch_wall_z"] == pytest.approx(
        cv15.AIR_BELOW + cv15.H_SUB, abs=1e-12)
    assert stack_check["n_sub_cells"] == cv15.N_SUB
    assert len(stack_check["eps_between"]) == cv15.N_SUB
    assert all(e == pytest.approx(cv15.EPS_R, abs=1e-6)
               for e in stack_check["eps_between"])
    # Recorded provenance, not the thing gated on (see _stack_check_ok).
    assert stack_check["ground_realization"] == "sheet"
    assert stack_check["patch_realization"] == "sheet"


def test_cv15_both_conductors_are_declared_sheets(capsys):
    """#931 §1.3: a sheet owns NO cell. Both conductors must come back in the
    ``pec_sheets`` collector on their declared node planes, and NEITHER may
    contribute a cell to ``pec_mask`` -- the substrate is the only body that
    occupies cells here, so if a conductor leaked into the cell mask the
    realized stack would gain a face the openEMS reference has no counterpart
    for. Checked on the collector, not on the wall planes, so the two
    assertions fail independently."""
    import numpy as np

    cv15 = _load_cv15()
    sim, grid, _ = _build_test_sim(cv15)
    pec_sheets: list = []
    pec_wires: list = []
    _mats, _d, _l, pec_mask, *_ = sim._assemble_materials(
        grid, pec_sheets=pec_sheets, pec_wires=pec_wires)
    capsys.readouterr()

    k_ground = grid.position_to_index(
        (cv15.DOM_X / 2, cv15.DOM_Y / 2, cv15.AIR_BELOW))[2]
    k_patch = grid.position_to_index(
        (cv15.DOM_X / 2, cv15.DOM_Y / 2, cv15.AIR_BELOW + cv15.H_SUB))[2]

    planes = sorted(sp.plane for sp in pec_sheets if sp.normal_axis == 2)
    assert planes == [k_ground, k_patch], (
        f"declared sheet planes {planes}, want [{k_ground}, {k_patch}]")
    assert not pec_wires
    # No PEC volume at all: pec_mask is either None or empty.
    assert pec_mask is None or not bool(np.asarray(pec_mask).any()), (
        "a declared sheet occupied cells -- it must own none (#931 §1.3)")


def test_cv15_no_wall_above_the_patch_plane(capsys):
    """The negative half of the stack check, asserted directly: over the patch
    footprint the realized z-wall planes are EXACTLY the two declared ones. A
    wall at ``k_patch + 1`` is the thing cv15 measured and rejected in 2026-08
    (11.9062 mm, no counterpart in the openEMS zero-thickness patch); before
    #931 its absence rested on a realization DEFAULT, and defaults are not
    evidence."""
    import numpy as np
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes

    cv15 = _load_cv15()
    sim, grid, _ = _build_test_sim(cv15)
    pec_sheets: list = []
    pec_wires: list = []
    _mats, _d, _l, pec_mask, *_ = sim._assemble_materials(
        grid, pec_sheets=pec_sheets, pec_wires=pec_wires)
    capsys.readouterr()
    periodic = sim._periodic_flags()
    edges = realized_pec_edge_masks(pec_mask, sheets=pec_sheets,
                                    wires=pec_wires, periodic=periodic)

    k_ground = grid.position_to_index(
        (cv15.DOM_X / 2, cv15.DOM_Y / 2, cv15.AIR_BELOW))[2]
    k_patch = grid.position_to_index(
        (cv15.DOM_X / 2, cv15.DOM_Y / 2, cv15.AIR_BELOW + cv15.H_SUB))[2]

    patch_fp = np.zeros(tuple(grid.shape)[:2], dtype=bool)
    for sp in pec_sheets:
        if sp.normal_axis == 2 and sp.plane == k_patch:
            patch_fp |= np.asarray(sp.footprint).any(axis=2)
    assert patch_fp.any()

    # Centre column of the patch, plus its four footprint rims -- the rim is
    # where a closed-vs-half-open footprint disagreement would show.
    ii, jj = np.nonzero(patch_fp)
    columns = {(int(ii.mean()), int(jj.mean())),
               (int(ii.min()), int(jj.min())), (int(ii.max()), int(jj.max())),
               (int(ii.min()), int(jj.max())), (int(ii.max()), int(jj.min()))}
    for (i, j) in columns:
        planes = set(realized_wall_planes(edges, 2, ij=(i, j), periodic=periodic))
        assert planes == {k_ground, k_patch}, (
            f"column ({i}, {j}) realizes z-wall planes {sorted(planes)}, "
            f"want exactly {{{k_ground}, {k_patch}}}")


def test_cv15_negative_control_ground_sheet_one_plane_low_raises(capsys):
    """NEGATIVE CONTROL 1 (issue #740 review, required change 5; #931 respelt):
    declare the ground sheet one node plane BELOW the substrate floor through
    the PRODUCTION builder and confirm the script's OWN
    ``assert_realized_stack`` -- not a test-local copy -- raises, naming
    z_sub_lo.

    This is the fail-before-fix witness. Before #740 this exact geometry -- an
    all-one-plane ground wall a cell below the floor -- silently passed every
    check in the file and produced ``rfx_one_plane_ground_b29f9de7.json``'s
    +6.09% vs openEMS. Under #931 it is reachable as a wrong DECLARATION
    rather than as a different realization rule for the same declaration,
    which is the point of the contract.
    """
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(
        cv15, ground_plane_z=cv15.AIR_BELOW - cv15.DX)
    with pytest.raises(RuntimeError, match="z_sub_lo"):
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()


def test_cv15_negative_control_patch_as_one_cell_volume_raises(capsys):
    """NEGATIVE CONTROL 2 (#931): the pre-#931 patch spelling -- a one-cell PEC
    ``Box`` -- is a filled slab with BOTH faces under the contract, so it grows
    an unreferenced wall at ``k_patch + 1``. The check must refuse to quote f0
    and must NAME that plane, because the number a reader needs is which wall
    appeared, not that something was wrong.
    """
    cv15 = _load_cv15()
    sim, grid, patch_shape = _build_test_sim(cv15, patch_kind="volume_1cell")
    with pytest.raises(RuntimeError, match="one-cell VOLUME") as exc:
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()
    # 11.9062 mm = z_sub_hi + DX, the plane this file measured and rejected in
    # 2026-08; printed as physical z, with the CPML pad offset removed.
    assert "11.9062" in str(exc.value), str(exc.value)


def test_cv15_builder_rejects_an_unknown_patch_kind():
    """The falsifier arm is a declaration switch, not a free-text field: a
    typo'd value must raise rather than silently fall back to production."""
    cv15 = _load_cv15()
    with pytest.raises(ValueError, match="patch_kind"):
        cv15.build_rfx_sim(patch_kind="one_plane")


# ---------------------------------------------------------------------------
# compare()'s stack-geometry gate: pinned with synthetic dicts (no solve),
# following test_crossval_gate_logic.py's precedent for this directory.
# ---------------------------------------------------------------------------

def _good_stack_check(cv15):
    return dict(
        ground_wall_z=cv15.AIR_BELOW,
        patch_wall_z=cv15.AIR_BELOW + cv15.H_SUB,
        n_sub_cells=cv15.N_SUB,
        eps_between=[cv15.EPS_R] * cv15.N_SUB,
        n_distinct_eps=cv15.N_DISTINCT_EPS_EXPECTED,
        ground_realization="sheet",
        patch_realization="sheet",
    )


def test_stack_check_ok_accepts_matching_measurement():
    cv15 = _load_cv15()
    ok, detail = cv15._stack_check_ok(_good_stack_check(cv15))
    assert ok, detail


def test_stack_check_ok_rejects_missing_leg():
    """A leg from before the #740 fix has no `stack_check` key at all --
    that must FAIL, not be skipped (the whole #740 defect was a leg that
    looked fine without this check)."""
    cv15 = _load_cv15()
    ok, detail = cv15._stack_check_ok(None)
    assert not ok
    assert "missing" in detail


def test_stack_check_ok_rejects_displaced_ground_wall():
    """The pre-fix one-plane-ground defect itself: ground wall one cell
    below z_sub_lo must FAIL even if n_sub_cells/eps happen to look right."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["ground_wall_z"] = cv15.AIR_BELOW - cv15.DX
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_wrong_eps_between():
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["eps_between"] = [1.0] * cv15.N_SUB  # vacuum, not the declared laminate
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_a_leg_that_never_measured_the_materials():
    """#931: ``n_distinct_eps`` is the witness that the sheets wrote no
    material (the deleted #702 own-cell resample). A leg recorded before that
    key existed FAILS -- same reasoning as a missing ``stack_check``: the
    property was not measured, so it is not evidence. This is what makes the
    committed pre-#931 ``rfx.json`` fail compare() until it is regenerated,
    rather than passing on a stack nobody checked."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    del sc["n_distinct_eps"]
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail
    assert "n_distinct_eps" in detail


def test_stack_check_ok_rejects_a_third_material():
    """A third distinct eps means something re-sampled a conductor's own cell
    or a partial fill appeared -- either way the cavity is not the declared
    stack."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["n_distinct_eps"] = cv15.N_DISTINCT_EPS_EXPECTED + 1
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_ignores_realization_label():
    """required change 1: ``ground_realization``/``patch_realization`` are
    recorded PROVENANCE only. A leg whose walls are correct but whose label
    says something else (a different mechanism landed the same planes) must
    still PASS."""
    cv15 = _load_cv15()
    sc = _good_stack_check(cv15)
    sc["ground_realization"] = "some_future_mechanism"
    sc["patch_realization"] = "some_future_mechanism"
    ok, detail = cv15._stack_check_ok(sc)
    assert ok, detail
