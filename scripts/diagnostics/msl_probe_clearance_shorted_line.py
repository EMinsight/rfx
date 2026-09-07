"""Probe-clearance bias against an EXACT truth: a shorted microstrip (#726).

The notch-filter attempt at this measurement failed its settling witness in
both arms, so nothing could be read from it. This fixture removes every
confound:

* the structure is a uniform microstrip terminated in a PEC short. At the
  short a lossless passive structure reflects everything, so |S11| = 1
  (0 dB) at EVERY frequency, not just at one notch — the truth is exact,
  frequency-independent, and needs no analytic model of the discontinuity;
* a short is a strong reflector, i.e. the standing-wave regime the board's
  divider puts the probes in;
* the only variable between arms is the distance from the port (and its
  probe comb) to the short.

Any deviation from 0 dB is extraction error. If the near arm deviates
materially more than the clean arm, preflight is right that probe
clearance biases |S11| and the Z0 guard's "the V*I-split S11/S21 are
unaffected" is wrong in this regime (issue #726).

RECORDED VERDICT (2026-08-28). Source: issue #726, comment of 2026-08-28 (the
same numbers are in docs/research_notes/20260817_cad_import_crossval_arc.md,
local to the primary checkout). short_x 20 mm, deepest probe to short
16.56 mm (clean) vs 0.16 mm (near), the same 5-probe comb in both arms:

  clean  settling -45.5 dB (bar -40: SETTLED). |S11| over 1.2-5.4 GHz:
         mean -0.04 dB, worst bin -0.23 dB, truth 0.00 dB. The extractor
         itself is accurate.
  near   settling witness -6.1 / -5.7 / -4.7 dB at num_periods 150 / 300 /
         600. Quadrupling the run moves it 1.4 dB the wrong way, so this
         arm cannot be READ at any run length. Ungated |S11| mean / worst
         in those records: -0.135 / -0.692, -0.084 / -0.712,
         -0.232 / -1.864 dB.

  Answer to the contradiction — a THIRD option, not either message:
  * preflight's "physical |S11| -> 1 may read as -5 to -10 dB" did NOT
    reproduce: even in the un-settleable records the worst bin stayed
    under 2 dB from the exact truth at every run length;
  * the guard's "the V*I-split S11/S21 are unaffected" is not reassurance
    either: the same condition makes the settling witness unsatisfiable,
    and settling is a precondition for quoting any DFT-derived S value.
  Observed: with the probes 0.16 mm from a PEC short the residual at the
  probes is independent of run length. The issue comment reads that as
  non-propagating content in the reflector's near field (the #388 class);
  this script does not instrument that mechanism.
  A board run whose port carries the clearance warning but whose settling
  witness reads -131 .. -157 dB is not in this regime; the settling number
  decides readability.
"""
from __future__ import annotations

import argparse
import math
import numpy as np

from rfx import Box, Simulation

C0 = 299792458.0
EPS_R = 3.66
H_SUB = 254e-6
W_TRACE = 600e-6
# #931 §1.3, off-lattice interfaces: DX was 80 um, which is 3.175 cells of
# substrate. The trace sheet then snaps to the nearest node — 240 um, 14 um
# INSIDE the laminate — and the realized board is not the declared one. The
# contract does not paper that over with a tie rule, so the mesh is redrawn
# on-lattice at H_SUB/3 (84.67 um, the rung nearest the old 80 um, so the cell
# count and the run cost barely move). Both faces are now exact node planes.
# This is a fixture change: the 2026-08-28 verdict recorded above was measured
# at 80 um and is a dated result, not a prediction for the re-run.
DX = H_SUB / 3
FREQ_MAX = 6e9


N_PROBE_OFFSET = 10      # cells; same comb in both arms
N_PROBE_SPACING = 2      # cells
N_PROBES = 5


def _assert_realized_stack(sim, x_probe, y_probe):
    """Build-time gate (#931), no solve: the realized z wall planes are the
    two declared ones — ground at z = 0 and trace at z = H_SUB — with no
    third plane anywhere. Under the pre-2.0 rule the ground's wall sat at
    z = -DX unless the drawing compensated for it, and that is exactly the
    class this fixture is meant to be free of.
    """
    from rfx.boundaries.pec import realized_pec_edge_masks, realized_wall_planes
    from rfx.geometry.rasterize_grid import coords_from_uniform_grid
    grid = sim._build_grid()
    ps: list = []
    pw: list = []
    _, _, _, pec_mask, *_ = sim._assemble_materials(
        grid, sheet_specs=[], pec_sheets=ps, pec_wires=pw)
    edges = realized_pec_edge_masks(pec_mask, sheets=ps, wires=pw,
                                   periodic=(False, False, False))
    coords = coords_from_uniform_grid(grid)
    nodes = np.asarray(coords.z)
    # a column on the trace, far from the shorting wall (which is a VOLUME and
    # stands a wall on every plane it spans, by design)
    i = int(np.argmin(np.abs(np.asarray(coords.x) - x_probe)))
    j = int(np.argmin(np.abs(np.asarray(coords.y) - y_probe)))
    got = sorted(float(nodes[k]) for k in realized_wall_planes(edges, 2, ij=(i, j))
                 if k < len(nodes))
    want = [0.0, H_SUB]
    if len(got) != 2 or any(abs(a - b) > 1e-12 for a, b in zip(got, want)):
        raise RuntimeError(
            "assert_realized_stack: declared ground z = 0.0000 mm and trace "
            f"z = {H_SUB * 1e3:.4f} mm, realized z wall planes "
            + ", ".join(f"{v * 1e3:.4f}" for v in got) + " mm")
    print(f"  realized z wall planes: {got[0]*1e3:.4f}, {got[1]*1e3:.4f} mm "
          "(declared ground 0.0000, trace "
          f"{H_SUB*1e3:.4f})  [#931 build-time gate, no solve]")


def build(line_len: float, port_x: float, short_x: float):
    margin = 2e-3
    LX = short_x + margin
    clearance = 2 * (2 * H_SUB + 8 * DX)
    LY = W_TRACE + 2 * clearance
    LZ = H_SUB + 1.5e-3
    sim = Simulation(freq_max=FREQ_MAX, domain=(LX, LY, LZ), dx=DX,
                     boundary="cpml", cpml_layers=8)
    sim.add_material("ro4350b", eps_r=EPS_R)
    sim.add(Box((0, 0, 0), (LX, LY, H_SUB)), material="ro4350b")
    y_c = LY / 2
    y_lo, y_hi = y_c - W_TRACE / 2, y_c + W_TRACE / 2
    # #931: printed copper is a SHEET on the substrate top plane. It used to
    # be drawn as a one-cell Box, whose single realized wall was its lo plane
    # — the same plane, by accident of the old rule.
    sim.add_thin_conductor(Box((0, y_lo, H_SUB), (short_x, y_hi, H_SUB)))
    # PEC short: a VOLUME wall from the ground plane up to the trace plane at
    # short_x. Under the contract it realizes tangential walls at BOTH z = 0
    # and z = H_SUB and shorts every normal edge between them, which is what
    # "short" means; the old drawing had to run one cell past the trace to get
    # one wall at the top.
    sim.add(Box((short_x - DX, y_lo, 0.0), (short_x, y_hi, H_SUB)),
            material="pec")
    # #931 migration rule 2: the ground plane used to be drawn ONE CELL BELOW
    # the board (`Box((0,0,-DX),(LX,LY,0))`) so that its only realized wall —
    # the lo node plane — landed at z = 0. That put the wall a cell below the
    # board it was meant to bound whenever the compensation was forgotten.
    # Declared as a sheet AT z = 0, no offset is needed.
    sim.add_thin_conductor(Box((0, 0, 0.0), (LX, LY, 0.0)))
    # BOTH arms use the SAME probe comb (offset/spacing/count) so the only
    # variable is the port's distance to the short. With the auto-resolved
    # comb the near arm's deepest probe landed PAST the short and outside
    # the domain (2026-08-27) — the comb geometry must be pinned, and its
    # extent checked, before the solve.
    sim.add_msl_port(position=(port_x, y_c, 0.0), width=W_TRACE, height=H_SUB,
                     direction="+x", impedance=50.0,
                     n_probe_offset=N_PROBE_OFFSET,
                     n_probe_spacing=N_PROBE_SPACING, n_probes=N_PROBES)
    _assert_realized_stack(sim, 0.5 * short_x, y_c)
    deepest = port_x + (N_PROBE_OFFSET + (N_PROBES - 1) * N_PROBE_SPACING) * DX
    if deepest >= short_x:
        raise SystemExit(
            f"design error: deepest probe {deepest * 1e3:.2f} mm is at or "
            f"past the short {short_x * 1e3:.2f} mm — the comb must fit "
            "between the port and the reflector")
    return sim, deepest


def run_arm(label, port_x, short_x, num_periods):
    sim, deepest = build(short_x, port_x, short_x)
    gap_mm = (short_x - deepest) * 1e3
    lam_g4 = C0 / (FREQ_MAX * math.sqrt(2.87)) / 4
    print(f"\n=== ARM {label}: port {port_x*1e3:.2f} mm, deepest probe "
          f"{deepest*1e3:.2f} mm, short {short_x*1e3:.2f} mm -> probe-to-short "
          f"gap {gap_mm:.2f} mm (lambda_g/4 at f_max = {lam_g4*1e3:.2f} mm) ===")
    adv = [str(a) for a in sim.preflight()]
    for a in adv:
        if "probe" in a and ("reflector" in a or "unsatisfiable" in a):
            print(f"  ! {a[:200]}")
    res = sim.compute_msl_s_matrix(n_freqs=60, num_periods=num_periods)
    f = np.asarray(res.freqs, float)
    S = np.abs(np.asarray(res.S[0, 0], np.complex128))
    sett = float(np.max(np.asarray(res.settling_db)))
    band = (f > 0.2 * FREQ_MAX) & (f < 0.9 * FREQ_MAX)
    db = 20 * np.log10(np.maximum(S[band], 1e-12))
    print(f"  settling witness: {sett:.1f} dB "
          f"({'SETTLED' if sett < -40 else 'NOT SETTLED — numbers not read'})")
    print(f"  |S11| over {0.2*FREQ_MAX/1e9:.1f}-{0.9*FREQ_MAX/1e9:.1f} GHz: "
          f"mean {db.mean():+.2f} dB, min {db.min():+.2f}, max {db.max():+.2f} "
          f"(truth: 0.00 dB at every bin)")
    return dict(settled=sett < -40, mean=db.mean(), worst=db.min(), sett=sett)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-periods", type=float, default=300.0)
    ap.add_argument("--short-x-mm", type=float, default=20.0)
    a = ap.parse_args()
    short_x = a.short_x_mm * 1e-3
    print("TRUTH: a shorted lossless line reflects everything -> |S11| = 0 dB "
          "at every frequency. Any deviation is extraction error.")
    clean = run_arm("clean", 2e-3, short_x, a.num_periods)
    if not clean["settled"]:
        # Precondition gate: without a settled control arm the comparison is
        # unreadable, so do not spend the second arm (2026-08-27: two
        # attempts were lost by running both arms first and reading after).
        print("\n=== VERDICT ===\n  NOT READ: control arm un-settled "
              f"({clean['sett']:.1f} dB, bar -40). Needed periods ~= "
              f"{a.num_periods * 40.0 / max(abs(clean['sett']), 1e-6):.0f}; "
              "rerun with --num-periods above that. Arm 2 skipped.")
        return
    near = run_arm("near", short_x - 1.6e-3, short_x, a.num_periods)
    print("\n=== VERDICT (pre-declared) ===")
    if not (clean["settled"] and near["settled"]):
        print("  NOT READ: a settling witness failed "
              f"(clean {clean['sett']:.1f} dB, near {near['sett']:.1f} dB). "
              "Raise --num-periods.")
        return
    d = abs(near["mean"]) - abs(clean["mean"])
    print(f"  |S11| mean error: clean {clean['mean']:+.2f} dB, "
          f"near {near['mean']:+.2f} dB (extra error {d:+.2f} dB)")
    print(f"  worst bin:        clean {clean['worst']:+.2f} dB, "
          f"near {near['worst']:+.2f} dB")
    if d >= 2.0:
        print("  -> PREFLIGHT right: probe clearance biases |S11| itself. The "
              "Z0 guard's 'V*I-split S11/S21 are unaffected' is wrong here.")
    else:
        print("  -> GUARD right within 2 dB: |S11| is insulated from the "
              "probe-clearance condition that wrecks Z0.")


if __name__ == "__main__":
    main()
