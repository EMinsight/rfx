"""Falsifier F1 replay for the #782 S11-gate re-pin — RETIRED BY #931, no FDTD.

WHAT IT WAS. It loaded the two saved arms from
``docs/design_notes/patch_edgefed_s11_band_repin_results.json`` (written by
``patch_edgefed_s11_band_repin.py``) and evaluated the COMMITTED gate's own
``_gate_readings`` + assertion conditions — imported from
``tests/locks/test_patch_edgefed_s11_passivity.py``, never re-implemented — on
each arm: the main arm had to pass every condition, and the retired arm had to
FAIL the in-band crossing (2b) and/or the antiresonance Re(Zin) floor (2c),
which is what showed the gate discriminated the bit-exact pre-#702 physics.

WHY IT NO LONGER RENDERS THAT VERDICT. The import is the point of the design and
also what retires it. Under the lattice ownership contract the gate's board was
redrawn with each foil on the laminate face it bounds, and its band was re-pinned
on the new board from VESSL 369367259226: ``RES_BAND_GHZ`` went (8.4, 9.2) ->
(7.4, 8.2). The frozen JSON is the OLD board, whose main-arm antiresonance
crossing is 8.8189 GHz. Evaluating today's band against that trace asks whether a
board that no longer exists resonates where a different board does; the answer is
no, and the honest reading of that "no" is that the question is void — the ~2-point
error class issue #782 documents, in its cross-board form.

So this script now DUMPS the frozen readings and says what they are, and does not
convert them into a pass/fail. It exits 0: a dated record is not a gate.

The predeclaration F1 stood for is DISCHARGED, not unmet — the #702 re-sample it
tested is deleted (design note #931 §2) and the geometry it compensated for is
drawn away. The live discrimination evidence is the re-pin runs (369367259225
Board H / 369367259226 Board S), not this file.
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import numpy as np  # noqa: E402

from tests.locks.test_patch_edgefed_s11_passivity import (  # noqa: E402
    PASSIVE_TOL, RES_BAND_GHZ, RES_BAND_RE_ZIN_MIN_OHM, RES_BAND_S11_MIN,
    _gate_readings,
)


def evaluate(arm: dict) -> dict:
    fr = np.asarray(arm["freqs_ghz"], dtype=float)
    s = np.asarray(arm["s11_re"], dtype=float) + 1j * np.asarray(arm["s11_im"], dtype=float)
    z0 = np.asarray(arm["z0_re"], dtype=float) + 1j * np.asarray(arm["z0_im"], dtype=float)
    g = _gate_readings(fr, s, z0)
    return dict(
        passivity=g["max_s11"] <= PASSIVE_TOL,
        band_floor=g["band_min_s11"] > RES_BAND_S11_MIN,
        band_crossing=bool(g["band_crossings_ghz"]),
        band_re_zin=g["band_max_re_zin"] > RES_BAND_RE_ZIN_MIN_OHM,
        dip_above_band=g["f_dip_ghz"] > RES_BAND_GHZ[1],
        readings=dict(
            max_s11=round(g["max_s11"], 4),
            band_min_s11=round(g["band_min_s11"], 4),
            band_max_re_zin=round(g["band_max_re_zin"], 1),
            crossings_ghz=[round(c, 4) for c in g["crossings_ghz"]],
            band_crossings_ghz=[round(c, 4) for c in g["band_crossings_ghz"]],
            f_dip_ghz=round(g["f_dip_ghz"], 4),
        ),
    )


def main() -> int:
    path = os.path.join(_REPO, "docs", "design_notes",
                        "patch_edgefed_s11_band_repin_results.json")
    with open(path) as f:
        results = json.load(f)
    print(f"[F1] evidence: {path}\n[F1] measured on tree {results['git_head']}\n"
          f"[F1] gate constants: band {RES_BAND_GHZ} GHz, floor {RES_BAND_S11_MIN}, "
          f"Re(Zin) > {RES_BAND_RE_ZIN_MIN_OHM} ohm, passivity {PASSIVE_TOL}")

    verdicts = {}
    for tag in ("main", "retired"):
        v = evaluate(results[tag])
        verdicts[tag] = v
        print(f"\n[F1] arm {tag} (bypass_resample={results[tag]['bypass_resample']}):")
        for k in ("passivity", "band_floor", "band_crossing", "band_re_zin",
                  "dip_above_band"):
            print(f"    {k:15s} {'PASS' if v[k] else 'FAIL'}")
        print(f"    readings: {v['readings']}")

    print(f"\n[F1] VERDICT: RETIRED (#931) — not computed.")
    print("[F1] The PASS/FAIL column above is today's committed band "
          f"{RES_BAND_GHZ} GHz read against a trace measured on the pre-#931 "
          "board, whose main-arm antiresonance crossing is 8.8189 GHz. The two "
          "describe different realized boards, so neither column is a verdict "
          "about either one; the readings are printed because they are dated "
          "evidence, not because they were checked.")
    print("[F1] The predeclaration is DISCHARGED: the #702 re-sample is deleted "
          "and the reserved-cell geometry is drawn away. Live discrimination "
          "evidence: VESSL 369367259225 (Board H) / 369367259226 (Board S).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
