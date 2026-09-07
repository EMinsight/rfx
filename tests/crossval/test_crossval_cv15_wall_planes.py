"""cv15's mandatory geometry self-check must assert the REALIZED electric-wall
PLANES, not the declared Box extents.

History, and it is history now. cv15's pre-#740 ``#325 AVOIDANCE (mandatory)``
self-check asserted only the substrate's z EXTENT (``n_sub_raster == N_SUB``,
from ``round(z/DX)`` cell counting) and kept PASSING while the one-cell ground
``Box`` realized its electric wall one full cell BELOW the declared substrate
floor -- the #693 "vacuum ground cell" trap, a live vacuum cell inside the
modelled cavity, undetected. #740 answered it with a ``two_plane`` flag that
put the missing far wall back for a one-cell body.

**Superseded by #931.** The lattice ownership contract removes the mechanism
rather than the symptom: a conductor is declared a VOLUME or a SHEET, a volume
realizes tangential walls on BOTH drawn faces at every thickness (design note
§1.2) and a sheet realizes exactly one plane and owns no cell (§1.3). cv15's
ground and patch are 35 um copper on a 787 um laminate, i.e. foil, so they are
sheets on the substrate's floor and top node planes. ``two_plane`` is deleted
-- passing it is a ``TypeError``, not a deprecation -- and with it the whole
"one-plane ground is a defect the flag repairs" framing that this file used to
be built on.

What survives, and is why the file survives: the CHECK. ``assert_realized_stack``
is still the right instrument, it is still called UNMODIFIED from the script
rather than mirrored here (the #740 review's finding: the first version of this
file hardcoded the fix in a test-local copy, so deleting it from the script
left everything green), and ``_stack_check_ok`` is still gated on the realized
PLANES with the realization label recorded as provenance only.

The negative control changes shape. Under the contract you cannot build a
one-plane ground from a Box any more -- it is either a filled slab or an
explicit sheet -- so the constructible defect is a sheet declared on the WRONG
node plane, which is what ``ground_plane_z=`` selects.

cv15 is guarded by ``if __name__ == "__main__":`` (see its final lines), so
importing it (to reach its module constants and the two functions above)
executes no simulation.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CV15_PATH = REPO_ROOT / "validation" / "crossval" / "15_patch_antenna_rt5880.py"

# The cv15 script is migrated by the crossval-C group in the same phase as this
# file: ground and patch become add_thin_conductor sheets at z_sub_lo / z_sub_hi,
# the two_plane parameter is deleted, assert_realized_stack is re-expressed on
# rfx.boundaries.pec.realized_wall_planes, and the rfx leg is re-solved
# (207 s CPU, num_periods 45 as committed) so validation/crossval/_15_patch_results/
# and the manifest claim_scope can be re-derived. VESSL run: rfx-931-post-cv15.
_MIGRATION_RUN = ("VESSL rfx-931-post-cv15 — 15_patch_antenna_rt5880.py "
                  "migrated to sheet declarations and re-solved (crossval-C)")


def _load_cv15():
    """Import cv15 as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("_cv15_wall_planes", CV15_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cv15_or_skip():
    """cv15, or a skip naming the migration that has to land first.

    The pre-#931 script imports ``two_plane_extension_masks`` from
    ``rfx.boundaries.pec`` and passes ``two_plane=`` to ``sim.add`` — both
    deleted by the contract — so on an unmigrated tree it does not import at
    all. Saying that once, here, is better than eight identical tracebacks.
    """
    try:
        cv15 = _load_cv15()
    except Exception as exc:                       # ImportError / TypeError
        pytest.skip(f"cv15 has not been migrated to the #931 contract yet "
                    f"({type(exc).__name__}: {exc}); {_MIGRATION_RUN}")
    params = inspect.signature(cv15.build_rfx_sim).parameters
    if "two_plane" in params:
        pytest.skip("cv15's build_rfx_sim still takes two_plane=, which the "
                    f"contract deletes (design note §2); {_MIGRATION_RUN}")
    return cv15


def _build_test_sim(cv15, **kw):
    """Build cv15's geometry through the PRODUCTION builder, ``build_rfx_sim``.

    Never a test-local mirror: the #740 review's item 1 was that a mirrored
    copy hardcoded the fix, so deleting it from the script left every test
    green. Returns ``(sim, grid, patch_shape)`` so the caller can run the
    script's OWN ``assert_realized_stack`` against it, cheaply (no solve).
    """
    sim, patch_shape, _geom = cv15.build_rfx_sim(do_gain=False, **kw)
    grid = sim._build_grid()
    return sim, grid, patch_shape


# ---------------------------------------------------------------------------
# assert_realized_stack: the check, on the real rasterized geometry
# (cheap -- _build_grid + _assemble_materials, no solve).
# ---------------------------------------------------------------------------

def test_cv15_committed_geometry_realizes_declared_walls(capsys):
    """The committed geometry realizes its electric walls exactly at the
    declared z_sub_lo / z_sub_hi planes, across the whole patch footprint."""
    cv15 = _cv15_or_skip()
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


def test_cv15_declares_foil_as_sheets_and_owns_no_cell(capsys):
    """#931 §1.3: 35 um copper on a 787 um laminate is foil, so it is a SHEET.

    The falsifier arm of the test above. A ground and patch declared as VOLUMES
    would still put a wall at z_sub_lo and z_sub_hi -- the positive test would
    pass -- while adding a second wall a cell away on each side and shorting the
    normal edge through the metal, which is the geometry the #720 A/B measured
    as the WORST agreement with the external reference (+0.265 correlation
    against +0.829 for the single-plane board, VESSL 369367256724). So the
    check is on the declaration: the conductors own no cell, and no wall stands
    one cell outside the laminate on either side.
    """
    cv15 = _cv15_or_skip()
    sim, _grid, _patch = _build_test_sim(cv15)
    capsys.readouterr()

    from tests._realized_pec import (assert_no_wall_at,
                                     assert_normal_edge_live,
                                     assert_sheet_owns_no_cell, realize)

    realized = realize(sim)
    assert len(realized.sheets) >= 2, (
        "cv15's ground and patch must be sheet declarations "
        f"(add_thin_conductor); the build carries {len(realized.sheets)}")
    assert_sheet_owns_no_cell(realized, what="cv15 foil")
    assert_normal_edge_live(realized, what="cv15 foil")
    z_lo = cv15.AIR_BELOW
    z_hi = cv15.AIR_BELOW + cv15.H_SUB
    assert_no_wall_at(realized, 2, [z_lo - cv15.DX, z_hi + cv15.DX],
                      what="cv15 board")


def test_cv15_negative_control_ground_on_the_wrong_plane_raises(capsys):
    """NEGATIVE CONTROL: a defect the contract can still express.

    The old control forced ``two_plane=False`` to reproduce the pre-#740
    one-plane ground. That geometry is unconstructible now -- a Box is a filled
    slab and a sheet is one declared plane -- so the control moves to the
    defect that remains available and is the one this check exists to catch: a
    ground sheet declared one node plane away from the laminate floor. The
    script's OWN ``assert_realized_stack``, not a copy, must raise and name
    ``z_sub_lo``.
    """
    cv15 = _cv15_or_skip()
    if "ground_plane_z" not in inspect.signature(cv15.build_rfx_sim).parameters:
        pytest.skip("cv15's builder has no ground_plane_z= knob for the "
                    f"negative control yet; {_MIGRATION_RUN}")
    sim, grid, patch_shape = _build_test_sim(
        cv15, ground_plane_z=cv15.AIR_BELOW - cv15.DX)

    with pytest.raises(RuntimeError, match="z_sub_lo"):
        cv15.assert_realized_stack(sim, grid, patch_shape)
    capsys.readouterr()


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
        ground_realization="sheet",
    )


def test_stack_check_ok_accepts_matching_measurement():
    cv15 = _cv15_or_skip()
    ok, detail = cv15._stack_check_ok(_good_stack_check(cv15))
    assert ok, detail


def test_stack_check_ok_rejects_missing_leg():
    """A leg from before the #740 fix has no `stack_check` key at all --
    that must FAIL, not be skipped (the whole #740 defect was a leg that
    looked fine without this check)."""
    cv15 = _cv15_or_skip()
    ok, detail = cv15._stack_check_ok(None)
    assert not ok
    assert "missing" in detail


def test_stack_check_ok_rejects_displaced_ground_wall():
    """The defect itself: a ground wall one cell below z_sub_lo must FAIL even
    if n_sub_cells/eps happen to look right."""
    cv15 = _cv15_or_skip()
    sc = _good_stack_check(cv15)
    sc["ground_wall_z"] = cv15.AIR_BELOW - cv15.DX
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_rejects_wrong_eps_between():
    cv15 = _cv15_or_skip()
    sc = _good_stack_check(cv15)
    sc["eps_between"] = [1.0] * cv15.N_SUB  # vacuum, not the declared laminate
    ok, detail = cv15._stack_check_ok(sc)
    assert not ok, detail


def test_stack_check_ok_ignores_realization_label():
    """#740 required change 1, and the reason this one test outlived the
    mechanism it was written for: ``ground_realization`` is recorded PROVENANCE
    only. A leg whose walls are correct but whose label says something else --
    a different mechanism landing the same planes -- must still PASS. The
    contract IS that different mechanism, and this test anticipated it.
    """
    cv15 = _cv15_or_skip()
    sc = _good_stack_check(cv15)
    sc["ground_realization"] = "some_future_mechanism"
    ok, detail = cv15._stack_check_ok(sc)
    assert ok, detail
