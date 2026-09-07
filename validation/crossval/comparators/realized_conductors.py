"""Build-time realized-conductor gate (lattice ownership contract, #931).

One place where every case in this group asks the SAME question of the SAME
function: *does the conductor the script drew realize where the script says
it does?*  No solve, no fields — grid construction plus material assembly
only, so a case can assert its geometry in well under a second and a
mis-realization is caught before any FDTD budget is spent.

The rule itself lives in ``rfx.boundaries.pec.realized_pec_edge_masks``
(design note ``docs/design_notes/20260906_plan_realign_lattice_ownership.md``
§1.7: one source, every consumer).  Nothing is re-derived here — this module
only asks the shared function and compares against what the caller declared.

Why a comparator module and not a per-case helper: the pre-#931 tree had a
private wall-plane derivation in cv15, another in preflight and another in
the MSL trace detector, and they disagreed.  The design-note answer is one
owner; the crossval-side answer is one gate, imported.

Vocabulary (design note §1):

* **volume** — a set of primal cells; every edge incident to an occupied cell
  is PEC, so a body drawn ``z_a -> z_b`` on node planes realizes tangential
  walls at BOTH planes;
* **sheet** — a footprint on ONE node plane, zero thickness; the in-plane
  edges whose two end nodes are both in the footprint are PEC and the normal
  edge stays live;
* **wire** — a 1-D lattice path of edges.

The falsifier discipline is copied from ``nu_cavity_gates``
(``extent_plus_one_fine_cell``): a gate that cannot fail by name on a
deliberately mis-realized geometry is not evidence.  :func:`assert_wall_planes`
raises with the declared and realized plane lists and their physical
coordinates, so the failure message IS the diagnosis.
"""

from __future__ import annotations

import numpy as np

from rfx.boundaries.pec import (
    realized_pec_edge_masks,
    realized_wall_planes,
)

__all__ = [
    "realize",
    "wall_planes",
    "sheet_planes",
    "footprint_nodes",
    "assert_wall_planes",
    "assert_no_conductor",
]


def _periodic_flags(sim) -> tuple[bool, bool, bool]:
    """The run's per-axis periodic flags (#689) — the ONE spelling.

    ``Simulation._periodic_flags`` is where the solver lanes read them, so
    this asks it rather than re-deriving from the boundary spec: a gate
    that realizes under different flags than the step function is not
    measuring the run.
    """
    return tuple(sim._periodic_flags())


def realize(sim, grid=None, *, lane: str = "uniform"):
    """``(edge_masks, sheets, wires, pec_mask, grid)`` for ``sim``.

    Build-time only: ``_build_grid`` (or ``_build_nonuniform_grid``) plus
    ``_assemble_materials``.  The sheet and wire collectors are passed, so a
    sheet-declared conductor is present — a caller that omits them gets a
    ``UserWarning`` and a silently conductor-free realization (design note §6
    out-parameters), which is exactly the failure this module exists to make
    impossible for the crossval scripts.
    """
    nu = lane == "nonuniform"
    if grid is None:
        grid = sim._build_nonuniform_grid() if nu else sim._build_grid()
    sheets: list = []
    wires: list = []
    assemble = sim._assemble_materials_nu if nu else sim._assemble_materials
    out = assemble(grid, pec_sheets=sheets, pec_wires=wires)
    pec_mask = out[3]
    if pec_mask is None and not sheets and not wires:
        # A conductor-free case. The shared function has nothing to read a
        # shape from and says so; a control still wants the empty answer,
        # so spell it here rather than teach the owner about emptiness.
        zero = np.zeros(tuple(grid.shape), dtype=bool)
        return (zero, zero.copy(), zero.copy()), sheets, wires, pec_mask, grid
    edges = realized_pec_edge_masks(
        pec_mask, sheets, wires, periodic=_periodic_flags(sim))
    return edges, sheets, wires, pec_mask, grid


def _node_coords(grid, axis: int):
    """Node coordinates along ``axis`` in metres, pad included."""
    from rfx.geometry.csg import _grid_coords
    return np.asarray(_grid_coords(grid)[axis], dtype=np.float64)


def wall_planes(sim, axis: int, *, at=None, region=None, grid=None,
                lane: str = "uniform"):
    """Realized tangential-wall node planes along ``axis``.

    ``at`` is a physical ``(u, v)`` pair on the two in-plane axes (in axis
    order, skipping ``axis``) naming ONE column; ``region`` is a tuple of
    three index slices.  Exactly one of the two, or neither for the whole
    grid.  Returns ``(planes, coords_m, grid)``.
    """
    edges, _sheets, _wires, _pm, grid = realize(sim, grid, lane=lane)
    kw = {}
    if at is not None:
        in_plane = [a for a in range(3) if a != axis]
        ij = []
        for a, u in zip(in_plane, at):
            nodes = _node_coords(grid, a)
            ij.append(int(np.argmin(np.abs(nodes - float(u)))))
        kw["ij"] = tuple(ij)
    elif region is not None:
        kw["region"] = tuple(region)
    planes = realized_wall_planes(
        edges, axis, periodic=_periodic_flags(sim), **kw)
    nodes = _node_coords(grid, axis)
    return planes, [float(nodes[k]) for k in planes], grid


def sheet_planes(sim, grid=None, *, lane: str = "uniform"):
    """``[(name, normal_axis, plane, coordinate_m), ...]`` for every declared
    sheet — what the SheetSpec says, before the edge rule is applied."""
    _edges, sheets, _wires, _pm, grid = realize(sim, grid, lane=lane)
    rows = []
    for sp in sheets:
        nodes = _node_coords(grid, sp.normal_axis)
        rows.append((sp.name, int(sp.normal_axis), int(sp.plane),
                     float(nodes[sp.plane])))
    return rows


def footprint_nodes(sim, axis: int, plane: int, *, component: int = 0,
                    along: int | None = None, at: int | None = None,
                    grid=None, lane: str = "uniform"):
    """Node indices of the realized ``component`` edges on ``(axis, plane)``.

    ``along`` is the axis to report node indices on and ``at`` the index on
    the remaining in-plane axis (defaults to that axis' mid-index).  Used to
    check a realized trace/patch width against the drawn one.
    """
    edges, _s, _w, _pm, grid = realize(sim, grid, lane=lane)
    m = np.asarray(edges[component], dtype=bool)
    in_plane = [a for a in range(3) if a != axis]
    if along is None:
        along = in_plane[0]
    other = [a for a in in_plane if a != along][0]
    if at is None:
        at = m.shape[other] // 2
    idx = [None, None, None]
    idx[axis] = int(plane)
    idx[other] = int(at)
    idx[along] = slice(None)
    line = m[tuple(idx)]
    return [int(v) for v in np.flatnonzero(line)]


def assert_wall_planes(sim, axis: int, declared, *, at=None, region=None,
                       grid=None, lane: str = "uniform", label: str = "conductor",
                       tol_m: float = 0.0):
    """Assert the realized wall planes along ``axis`` are exactly ``declared``.

    ``declared`` is either a sequence of node-plane INDICES or, when every
    entry is a float, a sequence of physical coordinates in metres (matched
    to the nearest node, then compared as indices — and additionally required
    to sit within ``tol_m`` of the declared coordinate when ``tol_m > 0``).

    Returns the realized ``{"planes", "coords_m"}`` record so the caller can
    log it beside its own numbers.  Raises ``AssertionError`` naming both
    lists on a mismatch — the cv24 ``extent`` falsifier design: a gate that
    fails BY NAME when the geometry is mis-realized by one plane.
    """
    planes, coords, grid = wall_planes(
        sim, axis, at=at, region=region, grid=grid, lane=lane)
    nodes = _node_coords(grid, axis)
    declared = list(declared)
    if declared and all(isinstance(v, float) for v in declared):
        want = [int(np.argmin(np.abs(nodes - v))) for v in declared]
        want_m = [float(v) for v in declared]
    else:
        want = [int(v) for v in declared]
        want_m = [float(nodes[k]) for k in want]
    if planes != sorted(want):
        raise AssertionError(
            f"[{label}] realized wall planes along {'xyz'[axis]} are {planes} "
            f"(z = {[round(c * 1e6, 4) for c in coords]} um) but the "
            f"declaration says {sorted(want)} "
            f"(= {[round(c * 1e6, 4) for c in want_m]} um). The conductor is "
            "not where the script says it is; fix the drawing or the "
            "declaration, never the gate (#931 §1.2/§1.3).")
    if tol_m > 0.0:
        for k, target in zip(planes, sorted(want_m)):
            off = abs(float(nodes[k]) - target)
            if off > tol_m:
                raise AssertionError(
                    f"[{label}] realized wall plane {k} sits at "
                    f"{nodes[k] * 1e6:.4f} um, {off * 1e6:.4f} um from the "
                    f"declared {target * 1e6:.4f} um (tol {tol_m * 1e6:.4f} um). "
                    "The declared plane is off-lattice; redraw the fixture on "
                    "the node line (#931 §1.3 off-lattice interfaces).")
    return {"planes": planes, "coords_m": coords}


def assert_no_conductor(sim, grid=None, *, lane: str = "uniform",
                        label: str = "control"):
    """Assert the case realizes NO conductor at all — no PEC cell, no sheet,
    no wire, no PEC edge.

    The dielectric controls (cv22, cv23, and every slab-rig arm) are the
    regression that #931 did not leak outside the conductor path: the
    contract changes PEC sampling only, and dielectric sampling (node,
    half-open) is untouched.  Asserting "no conductor" is what makes their
    bit-identity claim checkable rather than assumed.
    """
    edges, sheets, wires, pec_mask, grid = realize(sim, grid, lane=lane)
    n_cells = 0 if pec_mask is None else int(np.asarray(pec_mask).sum())
    n_edges = int(sum(np.asarray(m, dtype=bool).sum() for m in edges))
    if n_cells or sheets or wires or n_edges:
        raise AssertionError(
            f"[{label}] expected a conductor-free case but realized "
            f"{n_cells} PEC cells, {len(sheets)} sheets, {len(wires)} wires "
            f"and {n_edges} PEC E edges. Domain-boundary PEC is applied by "
            "the boundary spec and is NOT a body (#931 §1.8), so it must not "
            "appear here.")
    return {"pec_cells": 0, "sheets": 0, "wires": 0, "pec_edges": 0}
