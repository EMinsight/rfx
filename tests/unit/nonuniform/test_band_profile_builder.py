"""Pins for the axis-agnostic band profile builder (``make_band_profile``)
and the two builders rewired onto its engine (``make_z_profile``,
``_make_dz_profile``).

Pre-declaration (windows frozen before any code):
docs/design_notes/20260907_nu_band_profile_predeclaration.md. Falsifiers
F1-F5 and the (c) locks are pinned here; F6 is the battery rule, F7/F8 are
chain-model / FDTD witnesses in validation/research/multiband_nu/.

Revert-proof numbers, measured on main d990e18c BEFORE the fix (this
tree, PYTHONPATH pinned):

* ``_make_dz_profile`` on the 5-layer PCB stack (core 0.8 | prepreg 0.1 |
  core 0.8 | prepreg 0.1 | core 0.8 mm from z = 0.5 mm, domain 4.0 mm,
  dx 0.2 mm): nz 45, dz_min 8.333 um, max adjacent ratio **8.000** at
  index 30 (8.333 um next to 66.667 um), 25 ratios > 1.4 — a seam between
  two protected blocks that ``_smooth_preserving_blocks`` could never ramp.
* ``make_z_profile([1.0, 1.2, 2.5, 2.7] mm, 4 mm, 50 um, 200 um, 1.4)``:
  nz 32, ratio **4.527** at index 6 (226.36 -> 50 um), last cell 188.18 um
  (no descending ramp, contrary to its docstring).

Reverting either rewire reproduces those numbers and fails the pins below.
Profile level only — no FDTD.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rfx.auto_config import _make_dz_profile
from rfx.nonuniform import (
    interior_cells, make_band_profile, make_nonuniform_grid, make_z_profile,
)

ATOL = 1e-12          # I1 / I3: metres, never widened
RTOL_RATIO = 1e-9     # I2: absolute slack on the ratio, never widened

# --- fixtures ---------------------------------------------------------------

PCB_EDGES = [0.0, 0.5e-3, 1.3e-3, 1.4e-3, 2.2e-3, 2.3e-3, 3.1e-3, 4.0e-3]
PCB_FEATS = [(0.5e-3, 1.3e-3, 4.3), (1.3e-3, 1.4e-3, 4.3),
             (1.4e-3, 2.2e-3, 4.3), (2.2e-3, 2.3e-3, 4.3),
             (2.3e-3, 3.1e-3, 4.3)]
PCB_DX = 0.2e-3
PCB_DOMAIN = 4.0e-3

MZP_FEATS = [1.0e-3, 1.2e-3, 2.5e-3, 2.7e-3]
MZP_EDGES = [0.0] + MZP_FEATS + [4.0e-3]


def _nodes(cells):
    return np.concatenate([[0.0], np.cumsum(np.asarray(cells, dtype=float))])


def _ratios(cells):
    c = np.asarray(cells, dtype=float)
    if c.size < 2:
        return np.ones(0)
    r = c[1:] / c[:-1]
    return np.maximum(r, 1.0 / r)


def _iface_err(cells, edges):
    nodes = _nodes(cells)
    return max(float(np.min(np.abs(nodes - e))) for e in edges)


def _assert_invariants(cells, edges, cap, boundary_cell=None):
    cells = np.asarray(cells, dtype=float)
    assert cells.ndim == 1 and cells.dtype == np.float64
    assert np.all(cells > 0.0)
    err = _iface_err(cells, edges)
    assert err <= ATOL, f"I1: interface missed by {err:.3e} m"
    rr = _ratios(cells)
    if rr.size:
        assert float(rr.max()) <= cap + RTOL_RATIO, f"I2: ratio {rr.max():.9f} > {cap}"
    col = abs(float(np.sum(cells)) - (edges[-1] - edges[0]))
    assert col <= ATOL, f"I3: column off by {col:.3e} m"
    if boundary_cell is not None:
        assert cells[0] == boundary_cell and cells[-1] == boundary_cell, "I4"


def _block_index_ranges(cells, edges):
    nodes = _nodes(cells)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        i = int(np.argmin(np.abs(nodes - lo)))
        j = int(np.argmin(np.abs(nodes - hi)))
        out.append((i, j))
    return out


def _thirds_pair_indices(cells, block_ranges):
    """Adjacent-pair indices k (pair = cells[k], cells[k+1]) that are
    INTERNAL to a thirds split ([d/3, 2d/3, d] at a block's lo edge,
    [d, 2d/3, d/3] at its hi edge). R1: exempt from the ratio law."""
    c = np.asarray(cells, dtype=float)
    ex = set()
    for i, j in block_ranges:
        if j - i >= 3 and abs(c[i + 1] - 2.0 * c[i]) <= 1e-12 * c[i + 1]:
            ex.update((i, i + 1))
        if j - i >= 3 and abs(c[j - 2] - 2.0 * c[j - 1]) <= 1e-12 * c[j - 2]:
            ex.update((j - 3, j - 2))
    return ex


# --- (a) PCB 5-layer via the public builder ----------------------------------

def test_pcb_stack_builder_invariants_and_seam_rule():
    """F1-F3 on the PCB stack through ``make_band_profile`` (cap 1.4,
    min_cells 4, cores/prepregs protected). Revert-proof: the same stack
    through the old auto-z smoothing had seam ratio 8.000; here every
    seam, protected-to-protected included, is <= 1.4, and the coarser
    protected cores are refined (23 cells of 34.783 um) instead of a
    ramp being wedged into a 4-cell prepreg."""
    prot = [False, True, True, True, True, True, False]
    cells = make_band_profile(PCB_EDGES, [PCB_DX] * 7, max_ratio=1.4,
                              protected=prot, min_cells=4)
    _assert_invariants(cells, PCB_EDGES, 1.4)
    ranges = _block_index_ranges(cells, PCB_EDGES)
    counts = [j - i for i, j in ranges]
    # arithmetic of the seam rule: prepreg 4 x 25 um; core d <= 1.4 x 25 um
    # = 35 um -> ceil(0.8 mm / 35 um) = 23 cells of 34.783 um.
    assert counts[1:6] == [23, 4, 23, 4, 23], counts
    for k in (1, 2, 3, 4, 5):
        i, j = ranges[k]
        blk = cells[i:j]
        assert np.allclose(blk, blk[0], rtol=1e-13)
    assert abs(cells[ranges[1][0]] - (1.3e-3 - 0.5e-3) / 23) <= 1e-18
    # the core|prepreg seam on the actual seam cells: 34.783 / 25 um
    i, j = ranges[1]
    assert cells[j - 1] / cells[j] == pytest.approx((0.8e-3 / 23) / 25e-6, abs=1e-9)
    # the global maximum is the air ramp's even-spread ratio, under the cap
    assert float(_ratios(cells).max()) == pytest.approx(1.397368339, abs=1e-6)
    # protected blocks host no ramp cell: the count is exactly ceil(span/d)
    assert len(cells) == 89


# --- (a) PCB 5-layer via the auto-z path (R1: I2 outside thirds pairs) ------

def test_pcb_stack_auto_z_seam_rule_under_thirds():
    """F1-F3 on ``_make_dz_profile`` for the PCB stack. The thirds rule is
    kept as-is (it fires at every feature boundary, dielectric|dielectric
    included — #931 hand-off), so I2 is asserted on every pair OUTSIDE a
    thirds split at the auto path's 1.3, and every block-to-block seam is
    asserted on the ACTUAL seam cells (the 1/3 sub-cells).

    Revert-proof (main d990e18c): nz 45, seam ratio 8.000 at index 30,
    25 ratios > 1.4, cores 4-5-4 cells (float ceil on 8.000000000000002e-4).
    New: nz 115, dz_min 8.333 um unchanged, every seam 1.280, cores
    25 x 32 um each (+ their two thirds sub-cells): the prepreg's 8.333 um
    seam cell forces d_core/3 <= 1.3 x 8.333 um, d_core <= 32.5 um,
    ceil(0.8 mm / 32.5 um) = 25. The note's reference count of 23 was
    derived at cap 1.4; the auto path's frozen cap (F2, R6) is 1.3."""
    dz = np.asarray(_make_dz_profile(PCB_FEATS, PCB_DOMAIN, PCB_DX), dtype=float)
    assert _iface_err(dz, PCB_EDGES) <= ATOL
    assert abs(float(np.sum(dz)) - PCB_DOMAIN) <= ATOL
    ranges = _block_index_ranges(dz, PCB_EDGES)
    blocks = ranges[1:6]
    ex = _thirds_pair_indices(dz, blocks)
    rr = _ratios(dz)
    keep = np.array([k not in ex for k in range(rr.size)])
    assert float(rr[keep].max()) <= 1.3 + RTOL_RATIO, rr[keep].max()
    # block-to-block seams on the actual seam cells
    for (i0, j0), (i1, j1) in zip(blocks[:-1], blocks[1:]):
        assert j0 == i1
        a, b = dz[j0 - 1], dz[i1]
        assert max(a / b, b / a) <= 1.3 + RTOL_RATIO
    counts = [j - i for i, j in blocks]
    assert counts == [27, 6, 27, 6, 27], counts   # 25+2 / 4+2 splits
    assert len(dz) == 115
    assert float(np.min(dz)) == pytest.approx(0.1e-3 / 4 / 3, rel=1e-12)
    # prepreg block verbatim from apply_thirds_rule (unrefined); the
    # thickness is the builder's own float subtraction (1.0000000000000009e-4)
    d = (1.4e-3 - 1.3e-3) / 4
    i, j = blocks[1]
    assert np.array_equal(dz[i:j], np.array([d / 3, d * 2 / 3, d, d, d * 2 / 3, d / 3]))
    # refined cores: 25 x 32 um with the same split arithmetic
    dc = (1.3e-3 - 0.5e-3) / 25
    i, j = blocks[0]
    assert np.array_equal(
        dz[i:j], np.array([dc / 3, dc * 2 / 3] + [dc] * 23 + [dc * 2 / 3, dc / 3]))
    # R3: all three cores realize the same count
    assert counts[0] == counts[2] == counts[4]


# --- (b) make_z_profile ----------------------------------------------------

def test_make_z_profile_fine_coarse_fine_invariants():
    """F1-F3 on the Defect-2 fixture. Revert-proof: the old loop emitted
    ratio 4.527 at index 6 (226.36 -> 50 um) and ended on 188.18 um.
    Now every segment is fine -> coarse -> fine with the fine cell exactly
    ``dx_fine`` at both ends of every graded segment, every ratio
    <= 1.4, every feature on a node."""
    z = make_z_profile(MZP_FEATS, 4.0e-3, 50e-6, 200e-6, 1.4)
    _assert_invariants(z, MZP_EDGES, 1.4)
    assert z[0] == 50e-6 and z[-1] == 50e-6
    assert float(z.max()) <= 200e-6 * (1 + 1e-12)
    nodes = _nodes(z)
    for f in MZP_FEATS:
        k = int(np.argmin(np.abs(nodes - f)))
        assert z[k - 1] <= 50e-6 * (1 + 1e-12) and z[k] <= 50e-6 * (1 + 1e-12)
    assert float(_ratios(z).max()) == pytest.approx(1.352395022, abs=1e-6)
    assert len(z) == 42


def test_make_z_profile_no_grading_is_uniform_fine():
    z = make_z_profile([1.0e-3], 3.0e-3, 0.25e-3)
    assert np.allclose(z, 0.25e-3, rtol=1e-12)
    assert len(z) == 12
    _assert_invariants(z, [0.0, 1.0e-3, 3.0e-3], 1.4)


# --- (c) #763 locks stay: see test_auto_dz_profile_preserve.py ---------------
# (demo block bit-identical, dz_min 21.167 um, column 1.754 mm, air-run
# ratio <= 1.301; generic fixture interfaces + blocks bit-identical).

# --- (d) seeded random-stack fuzz -------------------------------------------

def _fuzz_family(n_stacks: int, seed: int = 20260907):
    """1-8 layers, thicknesses 20 um-3 mm (log-uniform), per-segment
    targets 0.05-2x the segment, protected mix, cap in {1.2, 1.3, 1.4},
    boundary_cell on for about half the stacks. Pinned-stack feasibility
    (refines the note's R4 "span >= 4 bc"): end segments free, their target
    equal to the pin, and bc <= span x min(1/4, (cap-1)/(1.1 cap)) — a
    descent from the pin at ratio cap needs up to bc/(cap-1) of span when
    the neighbour is much finer, and 4 bc only covers that for cap >= 1.34."""
    rng = np.random.default_rng(seed)
    for _ in range(n_stacks):
        nl = int(rng.integers(1, 9))
        th = np.exp(rng.uniform(np.log(20e-6), np.log(3e-3), size=nl))
        edges = np.concatenate([[0.0], np.cumsum(th)])
        tgt = th * rng.uniform(0.05, 2.0, size=nl)
        prot = rng.random(nl) < 0.5
        cap = float(rng.choice([1.2, 1.3, 1.4]))
        bc = None
        if rng.random() < 0.5:
            prot[0] = False
            prot[-1] = False
            lim = min(th[0], th[-1]) * min(0.25, (cap - 1.0) / (1.1 * cap))
            bc = float(min(tgt[0], tgt[-1], lim))
            tgt[0] = bc
            tgt[-1] = bc
        mc = int(rng.integers(1, 5))
        yield edges, tgt, prot, cap, bc, mc


def test_fuzz_random_stacks_invariants():
    """F1-F4 on >= 200 seeded stacks (rng 20260907), plus: every protected
    segment uniform with >= min_cells cells. Cell counts are reported in
    the design note, not gated."""
    n_ok = 0
    for edges, tgt, prot, cap, bc, mc in _fuzz_family(400):
        cells = make_band_profile(edges, tgt, max_ratio=cap, protected=prot,
                                  boundary_cell=bc, min_cells=mc)
        _assert_invariants(cells, edges, cap, boundary_cell=bc)
        for k, (i, j) in enumerate(_block_index_ranges(cells, edges)):
            if prot[k]:
                blk = cells[i:j]
                assert len(blk) >= mc
                assert np.allclose(blk, blk[0], rtol=1e-12)
        n_ok += 1
    assert n_ok == 400


# --- F5: axis round trip ----------------------------------------------------

F5_EDGES = [0.0, 12e-3, 15e-3, 27e-3]


def _f5_profile():
    return make_band_profile(F5_EDGES, [1e-3, 0.5e-3, 1e-3], max_ratio=1.3,
                             boundary_cell=1e-3)


def test_f5_profile_pins_and_invariants():
    p = _f5_profile()
    _assert_invariants(p, F5_EDGES, 1.3, boundary_cell=1e-3)


def test_f5_dx_dy_round_trip_no_ratio_warning_and_exact_extents():
    """The same profile as dx_profile AND dy_profile: Simulation constructs
    with zero 'adjacent cell ratio' warnings, preflight emits no
    nu_grading_ratio_beyond_validated_cap, I4 holds bit-exactly, and the
    grid's own interior node coordinates span 27 mm to 1e-12 m on both
    axes. nu_grading_reaches_absorber is reported (R7), not gated."""
    from rfx import Simulation
    p = _f5_profile()
    L = float(np.sum(p))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=10e9, domain=(L, L, 10e-3), dx=1e-3,
                         dx_profile=p.copy(), dy_profile=p.copy(),
                         boundary="cpml", cpml_layers=8)
    ratio_w = [x for x in w if "adjacent cell ratio" in str(x.message)]
    assert not ratio_w, [str(x.message) for x in ratio_w]
    report = sim.preflight()
    codes = {getattr(i, "code", None) for i in report.issues}
    assert "nu_grading_ratio_beyond_validated_cap" not in codes
    assert np.asarray(sim._dx_profile)[0] == 1e-3 and np.asarray(sim._dx_profile)[-1] == 1e-3
    assert np.asarray(sim._dy_profile)[0] == 1e-3 and np.asarray(sim._dy_profile)[-1] == 1e-3

    grid = make_nonuniform_grid((L, L), np.full(10, 1e-3), 1e-3, cpml_layers=8,
                                dx_profile=p, dy_profile=p)
    for arr, plo, phi in ((grid.dx_arr_f64, grid.pad_x_lo, grid.pad_x_hi),
                          (grid.dy_arr_f64, grid.pad_y_lo, grid.pad_y_hi)):
        inner = interior_cells(np.asarray(arr, dtype=float), plo, phi)
        assert abs(float(np.sum(inner)) - 27e-3) <= ATOL
        assert inner[0] == 1e-3 and inner[-1] == 1e-3


def test_f5_same_profile_as_dz_is_clean():
    from rfx import Simulation
    p = _f5_profile()
    L = float(np.sum(p))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sim = Simulation(freq_max=10e9, domain=(10e-3, 10e-3, L), dx=1e-3,
                         dz_profile=p.copy(), boundary="cpml", cpml_layers=8)
    assert not [x for x in w if "adjacent cell ratio" in str(x.message)]
    report = sim.preflight()
    codes = {getattr(i, "code", None) for i in report.issues}
    assert "nu_grading_ratio_beyond_validated_cap" not in codes


# --- engine corners ---------------------------------------------------------

def test_two_adjacent_protected_segments_refine_coarser_side():
    """A seam between two protected segments cannot host a ramp: the
    coarser one is refined until the seam ratio <= cap; both interfaces
    stay exact and both blocks stay uniform."""
    edges = [0.0, 1.0e-3, 1.1e-3, 2.1e-3]
    cells = make_band_profile(edges, [0.25e-3, 0.025e-3, 0.25e-3],
                              protected=[True, True, True], max_ratio=1.4)
    _assert_invariants(cells, edges, 1.4)
    (i0, j0), (i1, j1), (i2, j2) = _block_index_ranges(cells, edges)
    assert j1 - i1 == 4                       # the fine block is untouched
    assert np.allclose(cells[i1:j1], 25e-6, rtol=1e-12)
    assert np.allclose(cells[i0:j0], cells[i0], rtol=1e-12)
    assert cells[i0] <= 1.4 * 25e-6 * (1 + 1e-12)
    assert j0 - i0 == int(np.ceil(1.0e-3 / (1.4 * 25e-6)))


def test_thin_free_segment_is_a_refinement_source():
    """R5: a 20 um free gap between two 3 mm protected layers at 150 um
    realizes one 20 um cell and forces both neighbours down to <= 28 um."""
    edges = [0.0, 3e-3, 3.02e-3, 6.02e-3]
    cells = make_band_profile(edges, [150e-6, 20e-6, 150e-6],
                              protected=[True, False, True], max_ratio=1.4)
    _assert_invariants(cells, edges, 1.4)
    r = _block_index_ranges(cells, edges)
    assert r[1][1] - r[1][0] == 1 and abs(cells[r[1][0]] - 20e-6) <= ATOL
    assert cells[r[0][0]] <= 28e-6 * (1 + 1e-12)
    assert cells[r[2][0]] <= 28e-6 * (1 + 1e-12)


def test_free_segment_ramps_both_directions_around_a_fine_band():
    edges = [0.0, 10e-3, 11e-3, 21e-3]
    cells = make_band_profile(edges, [1e-3, 0.1e-3, 1e-3],
                              protected=[False, True, False], max_ratio=1.4)
    _assert_invariants(cells, edges, 1.4)
    (i0, j0), (i1, j1), (i2, j2) = _block_index_ranges(cells, edges)
    assert cells[j0 - 1] <= 0.14e-3 * (1 + 1e-12)      # descends into the band
    assert cells[i2] <= 0.14e-3 * (1 + 1e-12)          # ascends out of it
    assert float(cells.max()) <= 1e-3 * (1 + 1e-12)     # never above target


def test_boundary_cell_descending_ramp_from_pin():
    """A pin coarser than its segment's target: the run descends from the
    pin (the pin cannot move) and the far end is pinned too."""
    edges = [0.0, 10e-3, 20e-3]
    cells = make_band_profile(edges, [0.1e-3, 1e-3], max_ratio=1.4,
                              boundary_cell=1e-3)
    _assert_invariants(cells, edges, 1.4, boundary_cell=1e-3)
    assert cells[1] < cells[0] and cells[1] >= 1e-3 / 1.4 * (1 - 1e-12)


def test_boundary_cell_single_segment_exact_fit():
    cells = make_band_profile([0.0, 5e-3], [0.5e-3], max_ratio=1.3,
                              boundary_cell=0.5e-3)
    assert np.array_equal(cells, np.full(10, 0.5e-3))


def test_exact_multiple_realizes_uniform():
    cells = make_band_profile([0.0, 3e-3], [1e-3])
    assert np.array_equal(cells, np.full(3, 1e-3))


def test_protected_min_cells_and_ceil_tolerance():
    """R3: a float-dust quotient (8.000000000000002e-4 / 2e-4) realizes
    4 cells, not 5; min_cells lifts a thin block to the floor."""
    span = 2.2e-3 - 1.4e-3
    assert span / 0.2e-3 > 4.0
    cells = make_band_profile([0.0, span], [0.2e-3], protected=[True])
    assert len(cells) == 4
    cells = make_band_profile([0.0, 0.1e-3], [0.2e-3], protected=[True], min_cells=4)
    assert len(cells) == 4 and np.allclose(cells, 25e-6, rtol=1e-12)


@pytest.mark.parametrize("kwargs, match", [
    (dict(edges=[0.0, 1e-3], cell_sizes=[1e-4, 1e-4]), "len\\(edges\\)-1"),
    (dict(edges=[0.0, 1e-3, 0.5e-3], cell_sizes=[1e-4, 1e-4]), "strictly increasing"),
    (dict(edges=[0.0, 1e-3], cell_sizes=[0.0]), "> 0"),
    (dict(edges=[0.0, 1e-3], cell_sizes=[1e-4], max_ratio=1.0), "max_ratio"),
    (dict(edges=[0.0, 1e-3], cell_sizes=[1e-4], protected=[True],
          boundary_cell=1e-4), "FREE"),
    (dict(edges=[0.0, 1e-3], cell_sizes=[1e-4], boundary_cell=2e-3), "shorter"),
    (dict(edges=[0.0, 1e-3], cell_sizes=[1e-4], min_cells=0), "min_cells"),
])
def test_input_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_band_profile(**kwargs)


def test_pin_with_no_room_for_its_ramp_raises():
    """R4: a pinned end segment too short to descend from the pin to its
    fine protected neighbour is a contradiction, reported as ValueError."""
    with pytest.raises(ValueError):
        make_band_profile([0.0, 1.2e-3, 2.2e-3], [1e-3, 10e-6], max_ratio=1.2,
                          protected=[False, True], boundary_cell=1e-3)


def test_public_export():
    import rfx
    assert rfx.make_band_profile is make_band_profile
