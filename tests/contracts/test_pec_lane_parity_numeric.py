"""One geometry, every solver lane, one answer (#931 §1.7).

Stage B left a gap that no unit test could see: ``conductor_mask()``,
``fidelity_report()`` and the slice viewer all showed a declared PEC
SHEET, and no solver lane applied it — a sheet owns no cell, and every
solve entry took only ``pec_mask``. A simulation with a sheet-declared
ground plane ran as if the metal were not there.

So this file drives the actual lanes and compares FIELDS, not masks.

Battery, on a 20 mm PEC-walled box at dx = 1 mm with an Ez source below
the conductor and a probe above it:

* ``sheet``   — a zero-thickness PEC Box at z = 10 mm (§1.5);
* ``thin``    — the same rectangle through ``add_thin_conductor`` (PEC);
* ``volume``  — a 1-cell PEC Box from z = 10 mm to 11 mm;
* ``none``    — no conductor, the control.

``sheet`` and ``thin`` are the SAME object by the contract (G4 by
construction) and must agree to the bit on every lane. ``volume`` is a
different object — it shorts the normal edge between its two faces — and
must differ from ``sheet`` while still differing from ``none``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import rfx
from rfx import Box, Simulation

DX = 1e-3
DOMAIN = (20e-3, 20e-3, 20e-3)
Z_PLANE = 10e-3
SRC = (10e-3, 10e-3, 5e-3)
PROBE = (10e-3, 10e-3, 15e-3)
FOOT_LO = (4e-3, 4e-3)
FOOT_HI = (16e-3, 16e-3)

KINDS = ("none", "sheet", "thin", "volume")


def _build(kind, **sim_kw):
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec",
                     **sim_kw)
    lo = (FOOT_LO[0], FOOT_LO[1], Z_PLANE)
    hi = (FOOT_HI[0], FOOT_HI[1], Z_PLANE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "sheet":
            sim.add(Box(lo, hi), material="pec")
        elif kind == "thin":
            sim.add_thin_conductor(Box(lo, hi), sigma_bulk=5.8e7,
                                   thickness=1e-6)
        elif kind == "volume":
            sim.add(Box(lo, (hi[0], hi[1], Z_PLANE + DX)), material="pec")
    sim.add_source(position=SRC, component="ez")
    sim.add_probe(position=PROBE, component="ez")
    return sim


def _peak(sim, **run_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.run(n_steps=200, skip_preflight=True, **run_kw)
    return float(np.max(np.abs(np.asarray(res.time_series)[:, 0])))


def _traces_uniform():
    return {k: _peak(_build(k)) for k in KINDS}


def test_the_conductor_reaches_the_uniform_lane_at_all():
    """The stage-B gap: a declared sheet must change the field."""
    peaks = _traces_uniform()
    assert peaks["sheet"] != pytest.approx(peaks["none"], rel=1e-6), (
        "a declared PEC sheet did not reach the solver at all: "
        f"{peaks}")
    assert peaks["volume"] != pytest.approx(peaks["none"], rel=1e-6)


def test_sheet_and_pec_thin_conductor_are_the_same_object_on_the_uniform_lane():
    """G4 by construction (§1.3), measured on fields rather than masks."""
    peaks = _traces_uniform()
    assert peaks["sheet"] == pytest.approx(peaks["thin"], rel=0, abs=0), peaks


def test_a_one_cell_volume_is_not_the_same_object_as_a_sheet():
    """A volume shorts the normal edge between its two faces; a sheet does
    not (#690). They must not be silently equated."""
    peaks = _traces_uniform()
    assert peaks["volume"] != pytest.approx(peaks["sheet"], rel=1e-9), peaks


def test_the_nonuniform_lane_agrees_with_the_uniform_lane():
    """Same declarations on a uniform-valued NU mesh, same realization.

    The NU lane assembles, rasterizes and realizes through different code
    (``runners/nonuniform.py`` / ``rfx/nonuniform.py``); a uniform dz
    profile makes the two meshes the same lattice, so the CLASSIFICATION
    must agree even though the steppers differ numerically.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.runners.nonuniform import assemble_materials_nu

    for kind in ("sheet", "thin", "volume"):
        uni = _build(kind)
        grid_u = uni._build_grid()
        sheets_u: list = []
        _m, _d, _l, pec_u, *_ = uni._assemble_materials(
            grid_u, pec_sheets=sheets_u)

        nu = _build(kind, dz_profile=np.full(grid_u.nz - 1, DX))
        grid_n = nu._build_nonuniform_grid()
        sheets_n: list = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _mn, _dn, _ln, pec_n = assemble_materials_nu(
                nu, grid_n, pec_sheets=sheets_n)

        assert len(sheets_u) == len(sheets_n), kind
        if pec_u is None:
            assert pec_n is None, kind
        else:
            assert np.array_equal(np.asarray(pec_u), np.asarray(pec_n)), kind

        planes_u = realized_wall_planes(
            realized_pec_edge_masks(pec_u, sheets=tuple(sheets_u)), 2)
        planes_n = realized_wall_planes(
            realized_pec_edge_masks(pec_n, sheets=tuple(sheets_n)), 2)
        assert planes_u == planes_n, (kind, planes_u, planes_n)
        if kind == "volume":
            assert len(planes_u) == 2, (kind, planes_u)
        else:
            assert len(planes_u) == 1, (kind, planes_u)


def test_the_vmap_sweep_fast_path_realizes_the_same_conductor():
    """The batched lane builds its own scan body; it must see the sheet."""
    from rfx.vmap_sweep import vmap_material_sweep

    for kind in ("sheet", "volume"):
        sim = _build(kind)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            swept = vmap_material_sweep(sim, "eps_r", [1.0, 1.0], n_steps=200)
        ts = np.asarray(swept.time_series)
        batched = float(np.max(np.abs(ts[0, :, 0])))
        single = _peak(_build(kind))
        assert batched == pytest.approx(single, rel=2e-3), (kind, batched, single)


def test_the_adi_lane_realizes_the_same_conductor():
    """ADI zeroed E at the occupied CELL indices until #931."""
    peaks = {}
    for kind in ("none", "sheet", "volume"):
        sim = _build(kind, solver="adi")
        peaks[kind] = _peak(sim)
    assert peaks["sheet"] != pytest.approx(peaks["none"], rel=1e-6), peaks
    assert peaks["volume"] != pytest.approx(peaks["none"], rel=1e-6), peaks


def test_the_subgridded_lane_refuses_a_sheet_it_cannot_realize():
    """No silent drop: the SBP-SAT lane applies PEC from cell masks on two
    grids, so a sheet has no carrier there."""
    sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim.add(Box((FOOT_LO[0], FOOT_LO[1], Z_PLANE),
                    (FOOT_HI[0], FOOT_HI[1], Z_PLANE)), material="pec")
        sim.add_source(position=SRC, component="ez")
        sim.add_refinement(z_range=(6e-3, 14e-3), ratio=2)
        with pytest.raises(NotImplementedError, match="PEC sheets"):
            sim.run(n_steps=4, skip_preflight=True)
