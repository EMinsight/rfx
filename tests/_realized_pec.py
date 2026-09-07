"""Realized PEC geometry of a BUILT simulation — one reader, no solve (#931 §1.7).

Every crossval case that declares a conductor must be able to say, cheaply and
before any field is stepped, *where the solver actually put the metal*. Before
the lattice ownership contract each case answered that question for itself:
cv15 hand-rolled ``tangential_edge_masks`` plus a ``two_plane`` extension,
cv19 carried ``(t_c - 1)`` / ``(L_c + 1)`` arithmetic, cv18 an aperture ``- 1``,
cv09 a ``(2n - 1)·dx`` mirror identity, cv06b a ``_realized_trace_width``. Six
derivations, three of them mutually inconsistent about the same WR-90 iris and
the same microstrip trace.

This module is the single reader those cases now share. It calls
:func:`rfx.boundaries.pec.realized_pec_edge_masks` — the one function that
turns conductor geometry into PEC E edges — under the run's own periodic flags
(``Simulation._periodic_flags``), on the grid the run would actually build, and
reports the answer in the case's own physical units.

Scope: PEC BODIES (volumes, sheets, wires). Domain-boundary PEC/PMC faces are a
separate convention and stay out (design note §1.8): cv09, cv10, cv14 and cv24
gate on ``BoundarySpec`` faces and are unaffected by anything here.

Owner: the tests-crossval migration group (#931 phase 2, section T).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rfx.boundaries.pec import (
    realized_pec_edge_masks,
    realized_wall_planes,
)
from rfx.geometry.rasterize_grid import (
    coords_from_nonuniform_grid,
    coords_from_uniform_grid,
)
from rfx.nonuniform import NonUniformGrid

__all__ = [
    "RealizedPec",
    "realize",
    "nearest_plane",
    "wall_planes",
    "wall_positions",
    "assert_walls_at",
    "assert_no_wall_at",
    "assert_sheet_planes",
    "assert_sheet_owns_no_cell",
    "assert_normal_edge_live",
    "realized_extent",
]

_AXIS_NAME = ("x", "y", "z")


@dataclass(frozen=True)
class RealizedPec:
    """What the solver will realize for one built ``Simulation``.

    ``edges`` is the ``(Mx, My, Mz)`` triple: ``True`` where that E component
    is zeroed every step. ``cells`` is the VOLUME occupancy (``pec_mask``);
    sheets and wires own no cell, so they are absent from it by construction
    and present only in ``edges`` / ``sheets`` / ``wires``.
    """

    edges: tuple
    grid: object
    coords: object
    cells: object
    sheets: tuple
    wires: tuple
    materials: object
    periodic: tuple = (False, False, False)

    def nodes(self, axis: int) -> np.ndarray:
        """Node positions along ``axis`` in metres (host float64)."""
        return np.asarray((self.coords.x, self.coords.y, self.coords.z)[axis],
                          dtype=np.float64)


def _is_nonuniform(sim) -> bool:
    return any(getattr(sim, name, None) is not None
               for name in ("_dx_profile", "_dy_profile", "_dz_profile"))


def realize(sim, grid=None) -> RealizedPec:
    """Realize ``sim``'s conductors without stepping a single field.

    Uses the grid the run would build (non-uniform when any cell-size profile
    is set, uniform otherwise) and the run's own periodic flags, so the answer
    is the one the step function will apply — not the non-periodic default.
    """
    if grid is None:
        grid = (sim._build_nonuniform_grid() if _is_nonuniform(sim)
                else sim._build_grid())
    sheets: list = []
    wires: list = []
    if isinstance(grid, NonUniformGrid):
        materials, _, _, cells = sim._assemble_materials_nu(
            grid, pec_sheets=sheets, pec_wires=wires)
        coords = coords_from_nonuniform_grid(grid)
    else:
        materials, _, _, cells, _, _, _ = sim._assemble_materials(
            grid, pec_sheets=sheets, pec_wires=wires)
        coords = coords_from_uniform_grid(grid)
    if cells is None and not sheets and not wires:
        raise AssertionError(
            "realize(): this simulation declares no PEC body at all — no "
            "volume cells, no sheets, no wires. A case whose physics needs "
            "metal must not quote a result from a model that has none.")
    edges = realized_pec_edge_masks(
        cells, sheets=tuple(sheets), wires=tuple(wires),
        periodic=sim._periodic_flags())
    return RealizedPec(edges=edges, grid=grid, coords=coords, cells=cells,
                       sheets=tuple(sheets), wires=tuple(wires),
                       materials=materials, periodic=sim._periodic_flags())


def nearest_plane(realized: RealizedPec, axis: int, position: float,
                  *, tol_frac: float = 0.25) -> int:
    """Node index nearest ``position`` on ``axis``, refusing an off-lattice ask.

    A declared plane further than ``tol_frac`` of the local cell from any node
    is not a plane this lattice can carry: the case must be redrawn on-lattice
    (design note §1.3, "off-lattice interfaces"), not silently snapped.
    """
    nodes = realized.nodes(axis)
    k = int(np.argmin(np.abs(nodes - float(position))))
    gaps = np.diff(nodes)
    if gaps.size:
        d_local = float(gaps[min(k, gaps.size - 1)])
    else:                                             # pragma: no cover
        d_local = 1.0
    offset = abs(float(nodes[k]) - float(position))
    if offset > tol_frac * abs(d_local):
        raise AssertionError(
            f"declared plane {_AXIS_NAME[axis]} = {position:.9g} m is "
            f"{offset / abs(d_local):.3f} of a local cell ({d_local:.6g} m) "
            f"from the nearest node ({nodes[k]:.9g} m, index {k}). Redraw the "
            "case on-lattice; do not let it snap.")
    return k


def wall_planes(realized: RealizedPec, axis: int, *, region=None, ij=None):
    """Sorted node-plane indices along ``axis`` carrying a tangential wall."""
    return realized_wall_planes(realized.edges, axis, region=region, ij=ij,
                                periodic=realized.periodic)


def wall_positions(realized: RealizedPec, axis: int, *, region=None, ij=None):
    """:func:`wall_planes` in metres."""
    nodes = realized.nodes(axis)
    return [float(nodes[k]) for k in wall_planes(realized, axis, region=region,
                                                 ij=ij)]


def _footprint_mask(realized: RealizedPec, axis: int, footprint):
    """Normalize a footprint to a boolean mask over the two in-plane axes."""
    shape = tuple(realized.edges[0].shape)
    in_plane = tuple(a for a in range(3) if a != axis)
    want = (shape[in_plane[0]], shape[in_plane[1]])
    if footprint is None:
        return np.ones(want, dtype=bool)
    fp = np.asarray(footprint, dtype=bool)
    if fp.ndim == 3:
        fp = fp.any(axis=axis)
    if fp.shape != want:
        raise AssertionError(
            f"footprint shape {fp.shape} does not match the in-plane grid "
            f"{want} for a wall normal to {_AXIS_NAME[axis]}")
    return fp


def _full_wall(realized: RealizedPec, axis: int, k: int, footprint) -> bool:
    """True iff every tangential edge INSIDE the footprint is PEC on plane ``k``.

    "Inside" is the contract's own sentence (§1.3): the edge from node ``n`` to
    node ``n + 1`` belongs to the conductor iff BOTH of its end nodes are in
    the footprint. Asserting instead that every stored index under the
    footprint is PEC would demand an edge that leaves the footprint at its hi
    rim and would fail on a correctly realized sheet.

    The check is over the WHOLE footprint, not one column: cv15's #740 lesson
    is that a wall check evaluated at the feed passes while an in-plane
    rasterization edge case leaves a slit somewhere else.
    """
    fp = _footprint_mask(realized, axis, footprint)
    in_plane = tuple(a for a in range(3) if a != axis)
    for t in in_plane:
        m = np.asarray(realized.edges[t], dtype=bool)
        idx = [slice(None)] * 3
        idx[axis] = k
        plane = m[tuple(idx)]                     # (n_p, n_q) in in-plane order
        along = 0 if t == in_plane[0] else 1
        if along == 0:
            need = fp[:-1, :] & fp[1:, :]
            have = plane[:-1, :]
        else:
            need = fp[:, :-1] & fp[:, 1:]
            have = plane[:, :-1]
        if need.any() and not bool(np.all(have[need])):
            return False
    return True


def assert_walls_at(realized: RealizedPec, axis: int, positions,
                    *, footprint=None, what: str = "conductor") -> list:
    """Assert a tangential wall is realized at every declared ``position``.

    Returns the realized plane indices, so a caller can record what it
    measured rather than what it asked for.
    """
    out = []
    for position in positions:
        k = nearest_plane(realized, axis, position)
        if not _full_wall(realized, axis, k, footprint):
            planes = wall_positions(realized, axis)
            raise AssertionError(
                f"{what}: declared wall at {_AXIS_NAME[axis]} = "
                f"{position:.9g} m (node {k}) is NOT realized across its "
                f"footprint. Realized wall planes on this axis: {planes}.")
        out.append(k)
    return out


def assert_no_wall_at(realized: RealizedPec, axis: int, positions,
                      *, footprint=None, what: str = "conductor") -> None:
    """Assert NO tangential wall exists at each of ``positions``.

    The falsifier half of :func:`assert_walls_at`: a check that only ever
    looks for walls it wants cannot see an extra one, and an extra wall is
    exactly what a foil mis-declared as a volume produces.
    """
    for position in positions:
        k = nearest_plane(realized, axis, position)
        fp = _footprint_mask(realized, axis, footprint)
        for t in (a for a in range(3) if a != axis):
            m = np.asarray(realized.edges[t], dtype=bool)
            idx = [slice(None)] * 3
            idx[axis] = k
            if bool(np.any(m[tuple(idx)][fp])):
                raise AssertionError(
                    f"{what}: an UNDECLARED wall is realized at "
                    f"{_AXIS_NAME[axis]} = {position:.9g} m (node {k}), "
                    f"component E{_AXIS_NAME[t]}. Realized wall planes: "
                    f"{wall_positions(realized, axis)}.")


def assert_sheet_planes(realized: RealizedPec, axis: int, positions,
                        *, what: str = "sheet") -> list:
    """Assert the declared sheets land on exactly the declared node planes.

    Checks the ``SheetSpec`` list itself (a sheet's plane is a static int,
    never a traced quantity), so it fails on the declaration rather than on a
    downstream symptom.
    """
    want = sorted(nearest_plane(realized, axis, p) for p in positions)
    got = sorted(int(sp.plane) for sp in realized.sheets
                 if int(sp.normal_axis) == axis)
    if want != got:
        nodes = realized.nodes(axis)
        raise AssertionError(
            f"{what}: declared sheet planes {_AXIS_NAME[axis]} = "
            f"{[float(nodes[k]) for k in want]} m (nodes {want}) but the "
            f"build realized nodes {got} "
            f"({[float(nodes[k]) for k in got]} m).")
    return got


def realized_extent(realized: RealizedPec, axis: int, *, region=None,
                    ij=None) -> float:
    """Distance in metres between the outermost realized wall planes.

    This is the contract's answer to "how wide is the realized guide / how
    wide is the realized trace" (§1.9): the span between the bounding walls,
    not a node count and not ``round(W / dx) * dx``. Under the pre-#931 rule
    a body's far face was never a wall, so every consumer that measured a
    conductor this way read one cell too many on one side and one too few on
    the other — the #868 "40 mm guide reads 42 mm" case, and cv06b's
    ``635.0 um`` for a 600 um trace.
    """
    pos = wall_positions(realized, axis, region=region, ij=ij)
    if len(pos) < 2:
        raise AssertionError(
            f"realized_extent: axis {_AXIS_NAME[axis]} carries "
            f"{len(pos)} wall plane(s) {pos} — an extent needs two.")
    return float(pos[-1] - pos[0])


def assert_sheet_owns_no_cell(realized: RealizedPec, *,
                              what: str = "sheet") -> None:
    """The (S) half of the contract: a sheet adds no cell and no material.

    §1.3: "A sheet owns NO cell: it adds nothing to ``C``, writes no
    ``eps_r``/``sigma``, and is realized at exactly one plane." The direct
    replacement for the deleted #702 resample, which gave a one-node body
    its own cell's material at its live edge. Checked on the SheetSpec list
    and the assembled volume mask, so it fails on the declaration rather
    than on a downstream resonance shift.
    """
    if not realized.sheets:
        raise AssertionError(f"{what}: this build declares no sheet at all.")
    cells = realized.cells
    if cells is None:
        return
    occ = np.asarray(cells, dtype=bool)
    for sp in realized.sheets:
        a = int(sp.normal_axis)
        k = int(sp.plane)
        fp = np.asarray(sp.footprint, dtype=bool)
        idx = [slice(None)] * 3
        idx[a] = k
        overlap = occ[tuple(idx)] & fp[tuple(idx)]
        if bool(overlap.any()):
            raise AssertionError(
                f"{what} {sp.name!r}: its footprint on plane "
                f"{_AXIS_NAME[a]}={k} overlaps {int(overlap.sum())} VOLUME "
                "cell(s). A sheet owns no cell; a body that owns cells is a "
                "volume and must be declared with add().")


def assert_normal_edge_live(realized: RealizedPec, *, what: str = "sheet") -> None:
    """Normal E through a declared sheet stays live (§1.3, #690 semantics).

    The property that separates a sheet from a one-cell slab: a slab shorts
    the normal edge between its two faces, a sheet does not. Skipped for a
    sheet whose normal axis has length 1 (the 2-D lane), where the contract
    realizes the footprint as a 2-D volume and the "normal" component has no
    through-sheet edge.
    """
    if not realized.sheets:
        raise AssertionError(f"{what}: this build declares no sheet at all.")
    for sp in realized.sheets:
        a = int(sp.normal_axis)
        k = int(sp.plane)
        if realized.edges[a].shape[a] == 1:
            continue
        fp = np.asarray(sp.footprint, dtype=bool)
        m = np.asarray(realized.edges[a], dtype=bool)
        idx = [slice(None)] * 3
        idx[a] = k
        under = fp[tuple(idx)]
        if not under.any():                       # pragma: no cover
            continue
        if bool(np.all(m[tuple(idx)][under])):
            raise AssertionError(
                f"{what} {sp.name!r}: the normal component E{_AXIS_NAME[a]} is "
                f"zeroed everywhere under the footprint on plane "
                f"{_AXIS_NAME[a]}={k}. A sheet is zero-thickness — normal E "
                "through it stays live; a shorted normal edge means this was "
                "realized as a volume.")
