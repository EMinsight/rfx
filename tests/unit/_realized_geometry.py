"""Build-time (no solve) realization helpers for the lattice ownership
contract (#931).

Every migrated conductor in ``tests/unit/{nonuniform,runners,grid,subgrid}``
states, before any field is stepped, that what the lattice realizes is what
the fixture declares — read from the single owner
(``rfx.boundaries.pec.realized_pec_edge_masks`` /
``realized_wall_planes``), never from a Box mask and never from a number a
previous run happened to produce. Modelled on cv15's
``assert_realized_stack`` and cv24's ``extent`` gate.

The two things these helpers exist to hide from every call site:

* a Simulation carrying any of ``dx_profile`` / ``dy_profile`` /
  ``dz_profile`` must be assembled through ``_build_nonuniform_grid`` and
  ``_assemble_materials_nu``; getting that wrong silently assembles a
  DIFFERENT mesh from the one the run uses;
* ``_assemble_materials`` hands sheets and wires back through collector
  keywords, not in the positional tuple (design note §6). A caller that
  passes no collector does not get them, and a sheet-declared conductor
  then looks like an empty grid.
"""
from __future__ import annotations

import numpy as np


def realized(sim, grid=None):
    """``(grid, node_coords, edge_masks, sheets, pec_mask)`` for ``sim``.

    No solve, no run kwargs — this is the geometry the lanes will realize.
    """
    from rfx.nonuniform import NonUniformGrid
    from rfx.boundaries.pec import realized_pec_edge_masks
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)

    if grid is None:
        is_nu = (sim._dx_profile is not None or sim._dy_profile is not None
                 or sim._dz_profile is not None)
        grid = (sim._build_nonuniform_grid() if is_nu else sim._build_grid())
    sheets: list = []
    wires: list = []
    if isinstance(grid, NonUniformGrid):
        _, _, _, pec_mask = sim._assemble_materials_nu(
            grid, pec_sheets=sheets, pec_wires=wires)
        coords = coords_from_nonuniform_grid(grid)
    else:
        pec_mask = sim._assemble_materials(
            grid, pec_sheets=sheets, pec_wires=wires)[3]
        coords = coords_from_uniform_grid(grid)
    edges = realized_pec_edge_masks(pec_mask, sheets=sheets, wires=wires)
    return grid, coords, edges, sheets, pec_mask


def wall_positions(edges, coords, axis, **kw):
    """Realized wall planes along ``axis``, in metres."""
    from rfx.boundaries.pec import realized_wall_planes

    nodes = np.asarray((coords.x, coords.y, coords.z)[axis], dtype=float)
    return [float(nodes[k]) for k in realized_wall_planes(edges, axis, **kw)]


def assert_volume_spans(sim, lo, hi, *, axes=(0, 1, 2), atol=1e-9, grid=None):
    """A PEC VOLUME drawn between ``lo`` and ``hi`` realizes walls at BOTH
    drawn planes on every named axis, and nothing outside them (§1.2).

    ``lo`` / ``hi`` are 3-tuples or scalars (the same span on every axis).
    """
    lo3 = (lo,) * 3 if np.isscalar(lo) else tuple(lo)
    hi3 = (hi,) * 3 if np.isscalar(hi) else tuple(hi)
    grid, coords, edges, sheets, _ = realized(sim, grid)
    assert not sheets, "assert_volume_spans: this fixture declares a sheet"
    for axis in axes:
        pos = wall_positions(edges, coords, axis)
        assert pos, f"axis {'xyz'[axis]}: the conductor realizes NO wall plane"
        assert abs(min(pos) - lo3[axis]) < atol and abs(max(pos) - hi3[axis]) < atol, (
            f"axis {'xyz'[axis]}: realized walls span "
            f"[{min(pos) * 1e3:.4f}, {max(pos) * 1e3:.4f}] mm, drawn "
            f"[{lo3[axis] * 1e3:.4f}, {hi3[axis] * 1e3:.4f}] mm")
    return grid, coords, edges


def assert_sheet_plane(sim, z, *, normal_axis=2, n_sheets=1, atol=1e-9,
                       grid=None):
    """A declared SHEET realizes exactly one plane, at ``z``, owns no cell
    and leaves its normal E component live (§1.3, #690)."""
    grid, coords, edges, sheets, pec_mask = realized(sim, grid)
    assert len(sheets) == n_sheets, (
        f"expected {n_sheets} sheet declaration(s), got {len(sheets)}")
    for sp in sheets:
        assert sp.normal_axis == normal_axis
    pos = wall_positions(edges, coords, normal_axis)
    assert len(pos) == 1 and abs(pos[0] - z) < atol, (
        f"sheet realized at {[p * 1e3 for p in pos]} mm, declared "
        f"{z * 1e3:.4f} mm")
    if n_sheets == 1 and (pec_mask is None
                          or not bool(np.any(np.asarray(pec_mask)))):
        # only meaningful when the sheet is the ONLY conductor
        assert not bool(np.any(np.asarray(edges[normal_axis]))), (
            "a sheet leaves its normal E live; a one-cell volume shorts it")
    return grid, coords, edges
