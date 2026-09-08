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


# The 3-D ADI lane is unstable with interior PEC above unit CFL: measured on
# this battery at 200 steps, ``adi_cfl_factor=5`` (the constructor default)
# gives peak 4.1e+30 for the sheet and 6.9e+26 for the volume against a 0.0023
# control, and ``2.0`` gives 2.39 and 0.107 against 0.0043. So the lane tests
# below pin at ``1.0``, where all three kinds are finite. Without it "the
# conductor changed the answer" is satisfied by an overflow, which is not
# evidence that the conductor was realized.
ADI_STABLE_CFL = 1.0


def test_the_adi_lane_realizes_the_same_conductor():
    """ADI zeroed E at the occupied CELL indices until #931."""
    peaks = {}
    for kind in ("none", "sheet", "volume"):
        sim = _build(kind, solver="adi", adi_cfl_factor=ADI_STABLE_CFL)
        peaks[kind] = _peak(sim)
    assert all(np.isfinite(v) for v in peaks.values()), peaks
    assert peaks["sheet"] != pytest.approx(peaks["none"], rel=1e-6), peaks
    assert peaks["volume"] != pytest.approx(peaks["none"], rel=1e-6), peaks


def _forward_peak(sim, **fwd_kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sim.forward(n_steps=200, **fwd_kw)
    return float(np.max(np.abs(np.asarray(res.time_series)[:, 0])))


def test_the_adi_lane_realizes_the_same_conductor_through_forward_too():
    """``run()`` threaded the sheets and wires into the ADI lane; ``forward()``
    passed only ``pec_mask``, and a sheet owns no cell.

    Measured before the fix on a 20 x 20 mm 2-D TMz board at dx = 1 mm with a
    PEC sheet across the middle and the probe INSIDE it: ``run()`` read 0 and
    ``forward()`` read 0.0098782936 — bit-identical to the same model with the
    conductor deleted. The lane itself was never wrong;
    ``_run_adi_from_materials`` realizes sheets and wires through
    ``realized_pec_edge_masks``, so the whole defect was two arguments missing
    at one call site.

    The control matters: "the probe read zero" is also what a simulation that
    never ran reports, so ``none`` has to read nonzero on the same lane.
    """
    # "thin" is absent on purpose: ``add_thin_conductor`` is refused by name
    # on the ADI lane (`_validate_adi_configuration`), which is the §1.9
    # behaviour for a lane that cannot realize a declaration — pinned below.
    for kind in ("none", "sheet", "volume"):
        run_peak = _peak(_build(kind, solver="adi",
                                adi_cfl_factor=ADI_STABLE_CFL))
        fwd_peak = _forward_peak(_build(kind, solver="adi",
                                        adi_cfl_factor=ADI_STABLE_CFL))
        assert np.isfinite(run_peak) and np.isfinite(fwd_peak), (
            kind, run_peak, fwd_peak)
        assert fwd_peak == pytest.approx(run_peak, rel=1e-6), (
            f"{kind}: forward() {fwd_peak:.10g} disagrees with run() "
            f"{run_peak:.10g} on the ADI lane")

    control = _forward_peak(_build("none", solver="adi",
                                   adi_cfl_factor=ADI_STABLE_CFL))
    assert control > 0, "the control read zero — nothing ran"
    for kind in ("sheet", "volume"):
        peak = _forward_peak(_build(kind, solver="adi",
                                    adi_cfl_factor=ADI_STABLE_CFL))
        assert peak != pytest.approx(control, rel=1e-6), (
            f"{kind}: forward() on the ADI lane is indistinguishable from "
            "empty geometry")


def test_the_adi_forward_lane_refuses_the_thin_conductor_it_cannot_realize():
    """``add_thin_conductor`` has no carrier on the ADI lane, so both entry
    points refuse it by name rather than solving a board without its metal."""
    sim = _build("thin", solver="adi", adi_cfl_factor=ADI_STABLE_CFL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="thin-conductor"):
            sim.run(n_steps=4, skip_preflight=True)
        with pytest.raises(ValueError, match="thin-conductor"):
            _build("thin", solver="adi",
                   adi_cfl_factor=ADI_STABLE_CFL).forward(n_steps=4)


def test_the_adi_forward_lane_realizes_a_wire():
    """A filament owns no cell either, so it rode out of ``forward()`` on the
    same missing argument. Probed ON the wire's own Ez edge, with the wire
    removed as the control."""
    from rfx import PolylineWire

    def _wire_sim(with_wire):
        sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec",
                         solver="adi", adi_cfl_factor=ADI_STABLE_CFL)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if with_wire:
                sim.add(PolylineWire(((10e-3, 10e-3, 8e-3),
                                      (10e-3, 10e-3, 14e-3)), radius=0.2e-3),
                        material="pec")
        sim.add_source(position=SRC, component="ez")
        sim.add_probe(position=(10e-3, 10e-3, 11e-3), component="ez")
        return sim

    with_wire = _forward_peak(_wire_sim(True))
    without = _forward_peak(_wire_sim(False))
    assert np.isfinite(with_wire) and np.isfinite(without), (with_wire, without)
    assert without > 0, "the control read zero — nothing ran"
    assert with_wire != pytest.approx(without, rel=1e-6), (with_wire, without)


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


# ---------------------------------------------------------------------------
# the lossy (f0) sheet ctx on the forward lane (§1.7 one spelling, §1.9)
# ---------------------------------------------------------------------------

F0_DX = 2e-3
F0_DOM = (12e-3, 12e-3, 12e-3)


def test_the_forward_lossy_sheet_ctx_knows_about_a_pec_wire():
    """The lossy operator REPLACES the E update at its edges, so an edge a
    PEC conductor owns has to be removed from its ctx first.

    ``forward()`` asked "is there any PEC?" of ``pec_mask`` and the sheet list
    only. A filament owns no cell and is not a sheet, so a model whose only
    conductor is a wire reached the operator with ``pec_edge_masks=None`` and
    the operator wrote field back onto the wire's own PEC edge: measured, an
    Ex probe ON the filament read 0 through ``run()`` and 2.44e-6 through
    ``forward()``.
    """
    from rfx import PolylineWire

    def _build_wire():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(freq_max=15e9, domain=F0_DOM, dx=F0_DX,
                             boundary="pec")
            sim.add(PolylineWire(((4e-3, 6e-3, 6e-3), (8e-3, 6e-3, 6e-3)),
                                 radius=0.2e-3), material="pec")
            sim.add_thin_conductor(
                Box((2e-3, 2e-3, 6e-3), (10e-3, 10e-3, 6e-3)),
                sigma_bulk=5.8e7, surface_impedance_f0=10e9)
            sim.add_source(position=(4e-3, 4e-3, 2e-3), component="ez",
                           amplitude_kind="field")
            sim.add_probe(position=(5e-3, 6e-3, 6e-3), component="ex")
        return sim

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_ts = np.asarray(
            _build_wire().run(n_steps=60, skip_preflight=True).time_series)
        fwd_ts = np.asarray(_build_wire().forward(n_steps=60).time_series)
    run_peak = float(np.max(np.abs(run_ts[:, 0])))
    fwd_peak = float(np.max(np.abs(fwd_ts[:, 0])))
    assert run_peak == 0.0, run_peak
    assert fwd_peak == 0.0, (
        f"forward() left {fwd_peak:g} on a PEC filament's own edge: the lossy "
        "sheet ctx was built without the wires")


def test_the_forward_lossy_sheet_ctx_uses_the_runs_periodic_flags():
    """#689 on the OUTER call: the builder realizes the f0 footprint's own
    edges, and on a periodic axis the seam edge (node n-1 to node 0) is
    inside the sheet. ``forward()`` passed the flags to its PEC realization
    and not to the ctx, so the seam carried no loss — measured, 35 loaded Ex
    edges through ``run()`` against 30 through ``forward()`` on the same
    x-periodic board.
    """
    import rfx.materials.thin_conductor as _tc
    from rfx.boundaries.spec import Boundary, BoundarySpec

    def _build_periodic():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = Simulation(
                freq_max=15e9, domain=F0_DOM, dx=F0_DX,
                boundary=BoundarySpec(
                    x=Boundary(lo="periodic", hi="periodic"),
                    y=Boundary(lo="pec", hi="pec"),
                    z=Boundary(lo="pec", hi="pec")))
            sim.add_thin_conductor(
                Box((0.0, 2e-3, 6e-3), (12e-3, 10e-3, 6e-3)),
                sigma_bulk=5.8e7, surface_impedance_f0=10e9)
            sim.add_source(position=(4e-3, 4e-3, 2e-3), component="ez",
                           amplitude_kind="field")
            sim.add_probe(position=(4e-3, 6e-3, 6e-3), component="ex")
        return sim

    seen: list = []
    _orig = _tc.build_sheet_impedance_ctx

    def _record(*args, **kwargs):
        ctx = _orig(*args, **kwargs)
        seen.append(ctx)
        return ctx

    _tc.build_sheet_impedance_ctx = _record
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_res = _build_periodic().run(n_steps=60, skip_preflight=True)
            run_ctx = seen[-1]
            fwd_res = _build_periodic().forward(n_steps=60)
            fwd_ctx = seen[-1]
    finally:
        _tc.build_sheet_impedance_ctx = _orig

    for comp in ("mask_ex", "mask_ey", "mask_ez"):
        run_m = np.asarray(getattr(run_ctx, comp), dtype=bool)
        fwd_m = np.asarray(getattr(fwd_ctx, comp), dtype=bool)
        assert np.array_equal(run_m, fwd_m), (
            f"{comp}: run() loads {int(run_m.sum())} f0 edges, forward() "
            f"{int(fwd_m.sum())} — the ctx was built with different #689 flags")
    # the seam edge really is in the sheet, so the comparison has teeth
    seam = np.asarray(run_ctx.mask_ex, dtype=bool)[-1]
    assert seam.any(), "no seam edge in the loaded set — the pin is vacuous"

    run_peak = float(np.max(np.abs(np.asarray(run_res.time_series)[:, 0])))
    fwd_peak = float(np.max(np.abs(np.asarray(fwd_res.time_series)[:, 0])))
    assert fwd_peak == pytest.approx(run_peak, rel=1e-6), (run_peak, fwd_peak)
