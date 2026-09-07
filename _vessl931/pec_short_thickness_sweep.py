"""#931 T6 — the pre-declared separation for the PEC-short |S11| gate.

Two changes landed on `test_pec_short_s11_magnitude` together: the geometry
(the short was 0.93 of ONE cell against an unpinned dx, and §1.5 refuses that,
so it is redrawn as SHORT_CELLS whole cells on the node line) and the operator
(stage C made the waveguide S-matrix lane apply the realized PEC edges instead
of folding pec_mask into a sigma = 1e10 cell fill).

The separation, pre-declared in that module's docstring and in
docs/design_notes/931_migration/T6-RECOMPUTE.md before this ran:

    re-run at SHORT_CELLS = 1, 2, 4.
    If |S11| TRACKS the thickness it is the geometry, and the module re-pins
    itself from the thickness it declares.
    If |S11| does NOT move with thickness it is the operator, and it belongs
    with the chain-battery re-measure (T6-waveguide-chain-battery.md), not
    with a number in this module.

A total reflector's |S11| should not depend on how thick it is: everything
past the leading face is dark. So "tracks thickness" is itself a physical
statement about which mechanism is in play, not just a curve fit.

Run: JAX_PLATFORMS=cpu python _vessl931/pec_short_thickness_sweep.py
"""
from __future__ import annotations

import sys

import numpy as np

import tests.oracle.test_waveguide_port_validation_battery as B


def main() -> int:
    freqs = np.linspace(5.0e9, 7.0e9, 6)
    rows = []
    for cells in (1, 2, 4):
        B.SHORT_CELLS = cells
        sim = B._build_sim(freqs, pec_short_x=0.085,
                           waveform="modulated_gaussian")
        d = float(sim._build_grid().dx)
        faces = sim._pec_short_faces_m
        s, _, port_idx = B._s_matrix(sim, num_periods=40, normalize=False)
        s11 = np.abs(s[port_idx["left"], port_idx["left"], :])
        rows.append((cells, d, faces, s11))
        print(f"[pec-short-sweep] SHORT_CELLS={cells}  dx={d * 1e3:.4f} mm  "
              f"faces=({faces[0] * 1e3:.4f}, {faces[1] * 1e3:.4f}) mm  "
              f"thickness={(faces[1] - faces[0]) * 1e3:.4f} mm")
        print(f"[pec-short-sweep]   |S11| = "
              f"{np.array2string(s11, precision=5)}")
        print(f"[pec-short-sweep]   min={s11.min():.5f} mean={s11.mean():.5f} "
              f"max={s11.max():.5f}")
        sys.stdout.flush()

    mins = np.array([r[3].min() for r in rows])
    spread = float(mins.max() - mins.min())
    print("\n[pec-short-sweep] SUMMARY  min|S11| by SHORT_CELLS: "
          + ", ".join(f"{r[0]}->{r[3].min():.5f}" for r in rows))
    print(f"[pec-short-sweep] spread across thickness = {spread:.5f}")
    print("[pec-short-sweep] VERDICT: "
          + ("GEOMETRY — |S11| tracks thickness; the module re-pins from the "
             "declared thickness"
             if spread > 0.005 else
             "OPERATOR — |S11| is flat in thickness (spread <= 0.005), so the "
             "deficit is stage C's realized-edge lane, not the redraw; it "
             "belongs with the chain-battery re-measure"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
