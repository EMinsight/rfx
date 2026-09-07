"""TEMPORARY: pick the right lane for the shared realized-geometry check.

``tests/_realized_geometry`` is the single owner of the build-time
realization check under the lattice ownership contract (#931), and every
fixture in this branch reads the RULE from it. Two of its conveniences are
wrong on a non-uniform mesh, silently, and the four directories this shim
serves (``tests/unit/{nonuniform,runners,grid,subgrid}``) are where graded
meshes live:

* ``realized(sim)`` decides the lane from ``sim.dz_profile`` /
  ``sim._nu_axes``. ``Simulation`` has neither attribute — the fields are
  ``_dx_profile`` / ``_dy_profile`` / ``_dz_profile`` — so the probe never
  fires and a non-uniform simulation is assembled through ``_build_grid``
  and ``_assemble_materials`` while its run uses ``_build_nonuniform_grid``
  and ``_assemble_materials_nu``;
* ``node_index`` / ``_node_line`` have no ``NonUniformGrid`` branch.
  ``NonUniformGrid`` carries ``dx_arr`` / ``dy_arr`` / ``dz`` and no node
  line attribute, so the lookup falls back to ``(arange - pad) * grid.dx``,
  where on a graded axis ``dx`` is the BOUNDARY cell size.

Measured on ``test_inplane_grading_guards.py``'s fixture (a PEC Box drawn
3.0 -> 4.5 mm on ``dx_profile = dy_profile = [12x250um, 8x500um, 8x125um,
12x250um]``): the auto lane answers ``[18, 19, 20, 21, 22, 23, 24]`` and
the real lane answers ``[18, 19, 20, 21]``, while ``node_index`` on the
non-uniform grid maps 4.5 mm to 24. So the shared assertion would compare
planes from one mesh against indices from another.

This module fixes neither the rule nor the check — it states the lane and
reads the node line from the library's own spelling
(``coords_from_uniform_grid`` / ``coords_from_nonuniform_grid``, the same
functions the rasterizer uses). Delete it, and inline
``assert_wall_planes``, once the shared helper detects the lane.
Reported to the branch owner 2026-09-07.
"""
from __future__ import annotations

import numpy as np

__all__ = ["realized_on_lane", "wall_planes_m"]


def realized_on_lane(sim, *, nonuniform):
    """``tests._realized_geometry.realized`` with the lane stated."""
    from tests._realized_geometry import realized

    return realized(sim, nonuniform=bool(nonuniform))


def wall_planes_m(sim, axis, *, nonuniform, **kw):
    """``(Realization, [wall plane positions in metres])`` along ``axis``."""
    from rfx.geometry.rasterize_grid import (
        coords_from_nonuniform_grid, coords_from_uniform_grid)

    rz = realized_on_lane(sim, nonuniform=nonuniform)
    coords = (coords_from_nonuniform_grid(rz.grid) if nonuniform
              else coords_from_uniform_grid(rz.grid))
    nodes = np.asarray((coords.x, coords.y, coords.z)[axis], dtype=float)
    return rz, [float(nodes[k]) for k in rz.wall_planes(axis, **kw)]
