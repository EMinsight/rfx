"""Tests for 3D visualization module."""

import os
import pytest

import matplotlib
matplotlib.use("Agg")

from rfx import Simulation, Box, GaussianPulse
from rfx.visualize3d import (
    plot_geometry_3d, plot_field_3d, save_field_vtk, save_screenshot,
)


@pytest.fixture
def simple_sim():
    """A simple board: ground foil, substrate, trace foil.

    Ownership (#931 §1.5): ground and trace are foil, declared as SHEETS —
    zero-thickness Boxes on the substrate's two faces. Drawn one cell thick
    they were volumes, and the ground's cell then overlapped the substrate's,
    which is exactly the stack-up ambiguity #702 was about: under the contract
    a sheet owns no cell, so the substrate keeps the whole 0 .. 2 mm.
    """
    sim = Simulation(freq_max=5e9, domain=(0.02, 0.02, 0.01), dx=0.001)
    sim.add_material("substrate", eps_r=4.4, sigma=0.01)
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0)), material="pec")
    sim.add(Box((0, 0, 0), (0.02, 0.02, 0.002)), material="substrate")
    sim.add(Box((0.005, 0.005, 0.002), (0.015, 0.015, 0.002)), material="pec")
    sim.add_port(
        position=(0.01, 0.01, 0.001),
        component="ez",
        waveform=GaussianPulse(f0=3e9),
        extent=0.001,
    )
    return sim


def test_plot_geometry_3d_mpl(simple_sim):
    """plot_geometry_3d should produce a matplotlib figure."""
    fig = plot_geometry_3d(simple_sim, backend="matplotlib")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_plot_field_3d_mpl(simple_sim):
    """plot_field_3d should produce a figure from simulation state."""
    result = simple_sim.run(n_steps=50, compute_s_params=False)
    grid = simple_sim._build_grid()
    fig = plot_field_3d(result.state, grid, component="ez", backend="matplotlib")
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_save_field_vtk(simple_sim, tmp_path):
    """save_field_vtk should write a VTK file."""
    result = simple_sim.run(n_steps=50, compute_s_params=False)
    grid = simple_sim._build_grid()
    out = save_field_vtk(result.state, grid, str(tmp_path / "test_field"),
                         components=("ez",))
    assert os.path.exists(out)
    assert out.endswith(".vtk")
    # Check file is non-empty
    assert os.path.getsize(out) > 100


def test_save_screenshot(simple_sim, tmp_path):
    """save_screenshot should write a PNG file."""
    out = save_screenshot(simple_sim, filename=str(tmp_path / "test_geom"),
                          dpi=72)
    assert os.path.exists(out)
    assert out.endswith(".png")
    assert os.path.getsize(out) > 1000


def test_save_screenshot_with_field(simple_sim, tmp_path):
    """save_screenshot with state should write a PNG with field overlay."""
    result = simple_sim.run(n_steps=50, compute_s_params=False)
    out = save_screenshot(simple_sim, state=result.state,
                          filename=str(tmp_path / "test_field"),
                          dpi=72)
    assert os.path.exists(out)
    assert out.endswith(".png")


def test_the_two_foils_realize_as_sheets_and_leave_the_substrate_whole(simple_sim):
    """Build-time (no solve) ownership check for the fixture above (#931 §1.7).

    Two sheets on the substrate's two faces, no PEC cell anywhere, and the
    substrate's permittivity written at the ground plane too — which is the
    "a sheet owns no cell" clause read on a stack-up whose dielectric and
    metal were drawn overlapping.
    """
    import numpy as np
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes

    grid = simple_sim._build_grid()
    sheets: list = []
    mats, _, _, pec_mask, _, _, _ = simple_sim._assemble_materials(
        grid, pec_sheets=sheets)
    assert pec_mask is None, "foil declared as a sheet owns no cell"
    planes = sorted(sp.plane for sp in sheets)
    assert planes == [grid.position_to_index((0.0, 0.0, 0.0))[2],
                      grid.position_to_index((0.0, 0.0, 0.002))[2]]
    edges = realized_pec_edge_masks(pec_mask, sheets=sheets)
    assert realized_wall_planes(edges, 2) == planes
    i, j = grid.shape[0] // 2, grid.shape[1] // 2
    assert float(np.asarray(mats.eps_r)[i, j, planes[0]]) == pytest.approx(4.4)
