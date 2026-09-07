"""W6 — band-profile builder witnesses F7 (chain model, no FDTD) and F8
(FDTD, narrow fine band), pre-declared in
docs/design_notes/20260907_nu_band_profile_predeclaration.md §3.

F7: exact discrete scattering (``chain_model.scattering``) on the OLD
(main d990e18c ``_make_dz_profile``) and NEW auto-z PCB profiles embedded
in W2-count runways (140 lead cells of the profile's first cell, 150 tail
cells of its last). Gates: F7a ``|R|^2 <= 1.5 (sum_steps |R_step|)^2`` on
the NEW profile; F7b max non-thirds step ``<= |R_single(r=1.4, d=max P)|``.
The OLD profile is regenerated from the committed source of d990e18c
(``git show``), never retyped.

F8: 2-run differencing + geometric time gating (the W2 method,
``w2_w3_reflection.py``) on profile A(n_b) built BY THE BUILDER:
``[1.96 mm] x 140 | 1.4 | [1.0 mm] x n_b | 1.4 | [1.96 mm] x 150``, incident
from the coarse side, PEC-closed, TE10 soft Ex source at K_SRC = 85, probe
K_PRB = 100 (coarse cells). B run: ``[1.96] x 400 + [1.0] x 4`` (dt pin).
Gate: the n_b = 4 row, window ``|R_meas - R_model| <= 0.20 R_model + 3e-5``
(amplitude). Rows 2, 8, 16 trace the band-width law.

Usage (declared in the note):
    PYTHONPATH=. python -m validation.research.multiband_nu.w6_band_builder \
        --widths 2,4,8,16 --out validation/research/multiband_nu/results/w6_band_builder.json
    add ``--f7-only`` to skip the FDTD rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types

import numpy as np

import rfx
from rfx.auto_config import _make_dz_profile
from rfx.nonuniform import make_band_profile, make_nonuniform_grid, run_nonuniform

from . import fixtures as fx
from .chain_model import scattering
from .harness import build_pec_fixture
from .w2_w3_reflection import (
    F0, SIGMA_T, T0, FS2_FLOOR, dft_at, gaussian_sine, te10_sources, vg_of,
)

OLD_COMMIT = "d990e18ce0870ac7a893a86627e7232d1cf92c7b"

# --- F7 fixture -------------------------------------------------------------
PCB_FEATS = [(0.5e-3, 1.3e-3, 4.3), (1.3e-3, 1.4e-3, 4.3),
             (1.4e-3, 2.2e-3, 4.3), (2.2e-3, 2.3e-3, 4.3),
             (2.3e-3, 3.1e-3, 4.3)]
PCB_DOMAIN = 4.0e-3
PCB_DX = 0.2e-3
PCB_EDGES = [0.0, 0.5e-3, 1.3e-3, 1.4e-3, 2.2e-3, 2.3e-3, 3.1e-3, 4.0e-3]
F7_LEAD, F7_TAIL = 140, 150
F7_A = 4.5e-3            # transverse box; TE10 uses b = B_Y
F7_CAP_REF = 1.4


def _git_sha() -> str:
    """HEAD of the tree the run was made on (provenance, as W2/W4 record
    ``rfx.__file__``)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def _git_dirty() -> bool:
    """True when the worktree had uncommitted changes at run time."""
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], text=True) != ""
    except (subprocess.CalledProcessError, OSError):
        return True


def _old_make_dz_profile():
    """``_make_dz_profile`` exactly as committed on main d990e18c, loaded
    from ``git show`` into a throwaway module (provenance, not a retype)."""
    src = subprocess.check_output(
        ["git", "show", f"{OLD_COMMIT}:rfx/auto_config.py"], text=True)
    mod = types.ModuleType("_old_auto_config_d990e18c")
    mod.__file__ = f"git:{OLD_COMMIT}:rfx/auto_config.py"
    # the old module's @dataclass looks itself up in sys.modules
    sys.modules[mod.__name__] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod._make_dz_profile


def _embed(profile: np.ndarray) -> np.ndarray:
    p = np.asarray(profile, dtype=np.float64)
    return np.concatenate([np.full(F7_LEAD, p[0]), p, np.full(F7_TAIL, p[-1])])


def _thirds_pairs(profile: np.ndarray, edges) -> set[int]:
    c = np.asarray(profile, dtype=float)
    nodes = np.concatenate([[0.0], np.cumsum(c)])
    ex: set[int] = set()
    for lo, hi in zip(edges[1:-2], edges[2:-1]):    # the five blocks
        i = int(np.argmin(np.abs(nodes - lo)))
        j = int(np.argmin(np.abs(nodes - hi)))
        if j - i >= 3 and abs(c[i + 1] - 2 * c[i]) <= 1e-12 * c[i + 1]:
            ex.update((i, i + 1))
        if j - i >= 3 and abs(c[j - 2] - 2 * c[j - 1]) <= 1e-12 * c[j - 2]:
            ex.update((j - 3, j - 2))
    return ex


def f7_profile_report(name: str, profile: np.ndarray, dt: float, dy: float,
                      b: float) -> dict:
    emb = _embed(profile)
    R, T = scattering(emb, F7_LEAD, F7_TAIL, F0, dt, dy, b)
    steps = []
    ex = _thirds_pairs(profile, PCB_EDGES)
    for k in range(len(profile) - 1):
        d0, d1 = float(profile[k]), float(profile[k + 1])
        if d0 == d1:
            continue
        Rk, _ = scattering(np.array([d0] * F7_LEAD + [d1] * F7_TAIL), F7_LEAD,
                           F7_TAIL, F0, dt, dy, b)
        steps.append({"index": k, "d_from_um": d0 * 1e6, "d_to_um": d1 * 1e6,
                      "ratio": max(d0 / d1, d1 / d0), "R_step": abs(Rk),
                      "thirds_pair": k in ex})
    non_thirds = [s for s in steps if not s["thirds_pair"]]
    thirds = [s for s in steps if s["thirds_pair"]]
    sum_amp = sum(s["R_step"] for s in steps)
    sum_pow = sum(s["R_step"] ** 2 for s in steps)
    dmax = float(np.max(profile))
    R1, _ = scattering(np.array([dmax / F7_CAP_REF] * F7_LEAD + [dmax] * F7_TAIL),
                       F7_LEAD, F7_TAIL, F0, dt, dy, b)
    max_nt = max(s["R_step"] for s in non_thirds) if non_thirds else 0.0
    max_th = max(s["R_step"] for s in thirds) if thirds else 0.0
    rr = profile[1:] / profile[:-1]
    rr = np.maximum(rr, 1 / rr)
    return {
        "name": name, "nz": int(len(profile)),
        "dz_min_um": float(np.min(profile)) * 1e6,
        "dz_max_um": dmax * 1e6, "max_ratio": float(rr.max()),
        "n_ratio_over_1p4": int(np.sum(rr > 1.4 + 1e-9)),
        "R_total": abs(R), "R_total_sq": abs(R) ** 2, "T_total": abs(T),
        "n_steps": len(steps), "sum_abs_R_step": sum_amp,
        "sum_abs_R_step_sq": sum_amp ** 2, "sum_R_step_sq": sum_pow,
        "max_nonthirds_step": max_nt, "max_thirds_step": max_th,
        "R_single_cap_at_dmax": abs(R1),
        "f7a_window": 1.5 * sum_amp ** 2,
        "f7a_fired": bool(abs(R) ** 2 > 1.5 * sum_amp ** 2),
        "f7b_fired": bool(max_nt > abs(R1) * (1 + 1e-9)),
        "steps": steps, "cells_um": (profile * 1e6).tolist(),
    }


def run_f7() -> dict:
    old_fn = _old_make_dz_profile()
    old = np.asarray(old_fn(PCB_FEATS, PCB_DOMAIN, PCB_DX), dtype=np.float64)
    new = np.asarray(_make_dz_profile(PCB_FEATS, PCB_DOMAIN, PCB_DX), dtype=np.float64)
    grid_old = make_nonuniform_grid((F7_A, fx.B_Y), _embed(old), PCB_DX, cpml_layers=0)
    grid_new = make_nonuniform_grid((F7_A, fx.B_Y), _embed(new), PCB_DX, cpml_layers=0)
    dt_old, dt_new = float(grid_old.dt), float(grid_new.dt)
    dy = PCB_DX
    out = {"old_commit": OLD_COMMIT, "dt_old_s": dt_old, "dt_new_s": dt_new,
           "F0_Hz": F0, "dy_m": dy, "b_m": fx.B_Y,
           "lead_cells": F7_LEAD, "tail_cells": F7_TAIL}
    out["old"] = f7_profile_report("OLD (d990e18c _make_dz_profile)", old, dt_old, dy, fx.B_Y)
    out["new"] = f7_profile_report("NEW (_make_dz_profile on the band engine)", new, dt_new, dy, fx.B_Y)
    for key in ("old", "new"):
        r = out[key]
        print(f"F7 {r['name']}: nz={r['nz']} dz_min={r['dz_min_um']:.3f}um "
              f"max_ratio={r['max_ratio']:.3f} |R|={r['R_total']:.4e} "
              f"|R|^2={r['R_total_sq']:.4e} sum|Rstep|={r['sum_abs_R_step']:.4e} "
              f"(sum)^2={r['sum_abs_R_step_sq']:.4e} sum(Rstep^2)={r['sum_R_step_sq']:.4e} "
              f"max_nonthirds={r['max_nonthirds_step']:.4e} max_thirds={r['max_thirds_step']:.4e} "
              f"R_single(1.4,dmax)={r['R_single_cap_at_dmax']:.4e} "
              f"F7a_window={r['f7a_window']:.4e} fired={r['f7a_fired']} F7b_fired={r['f7b_fired']}",
              flush=True)
    return out


# --- F8 ----------------------------------------------------------------------
DC = fx.DZ_FINE * 1.4 ** 2         # 1.96 mm coarse cell
DR = fx.DZ_FINE * 1.4              # 1.4 mm ramp cell
N_LEAD_C = 140
N_TAIL_C = 150
K_SRC = 85
K_PRB = 100
N_STEPS = 1200
B_N_COARSE = 400
B_N_FINE_PIN = 4


def a_profile(n_b: int) -> np.ndarray:
    z1 = N_LEAD_C * DC + DR
    z2 = z1 + n_b * fx.DZ_FINE
    z3 = z2 + DR + N_TAIL_C * DC
    return make_band_profile([0.0, z1, z2, z3], [DC, fx.DZ_FINE, DC],
                             protected=[False, True, False], max_ratio=1.4)


def a_profile_expected(n_b: int) -> np.ndarray:
    return np.asarray([DC] * N_LEAD_C + [DR] + [fx.DZ_FINE] * n_b + [DR]
                      + [DC] * N_TAIL_C, dtype=np.float64)


def b_profile() -> np.ndarray:
    return np.asarray([DC] * B_N_COARSE + [fx.DZ_FINE] * B_N_FINE_PIN, dtype=np.float64)


def _run_probe(profile: np.ndarray, n_steps: int):
    grid, mats = build_pec_fixture(profile, (fx.A_X, fx.B_Y), fx.DXY)
    wf = gaussian_sine(n_steps, float(grid.dt), SIGMA_T, T0)
    srcs = te10_sources(grid, K_SRC, wf)
    out = run_nonuniform(grid, mats, n_steps, sources=srcs,
                         probes=[(1, grid.ny // 2, K_PRB, "ex")])
    return grid, np.asarray(out["time_series"][:, 0], dtype=np.float64)


def _cell_delay(profile: np.ndarray, dt: float, dy: float, b: float) -> float:
    cache: dict[float, float] = {}
    tot = 0.0
    for d in profile:
        d = float(d)
        if d not in cache:
            cache[d] = vg_of(d, dt, dy, b)
        tot += d / cache[d]
    return tot


def f8_arm(n_b: int, trace_b: np.ndarray, grid_b) -> dict:
    prof = a_profile(n_b)
    expected = a_profile_expected(n_b)
    builder_exact = (len(prof) == len(expected)
                     and bool(np.max(np.abs(prof - expected)) <= 1e-12))
    grid_a, trace_a = _run_probe(prof, len(trace_b))
    dt = float(grid_a.dt)
    assert abs(dt - float(grid_b.dt)) < 1e-20, (dt, float(grid_b.dt))
    dy, b = float(grid_a.dy), fx.B_Y
    vg_c = vg_of(DC, dt, dy, b)
    z_src, z_prb = K_SRC * DC, K_PRB * DC
    z_tr = N_LEAD_C * DC
    t_r = T0 + (2 * z_tr - z_src - z_prb) / vg_c
    t_s = T0 + (z_src + 2 * z_tr - z_prb) / vg_c
    # far wall: through the band + tail and back
    beyond = prof[N_LEAD_C:]
    t_f = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(beyond, dt, dy, b) + (z_tr - z_prb) / vg_c
    # last band-internal return: to the far ramp/band edge and back
    band_and_ramps = prof[N_LEAD_C:N_LEAD_C + 2 + n_b]
    t_band = T0 + (z_tr - z_src) / vg_c + 2 * _cell_delay(band_and_ramps, dt, dy, b) + (z_tr - z_prb) / vg_c
    gate_end = min(t_s, t_f) - 4 * SIGMA_T
    assert t_r + 4 * SIGMA_T < gate_end, (t_r, t_s, t_f)
    diff = trace_a - trace_b
    n_gate = int(gate_end / dt)
    refl = dft_at(diff, dt, F0, 0, n_gate)
    t_echo_prb = T0 + (z_src + z_prb) / vg_c
    t_inc_end = min(T0 + (z_prb - z_src) / vg_c + 8 * SIGMA_T, t_echo_prb - 4 * SIGMA_T)
    inc = dft_at(trace_b, dt, F0, 0, int(t_inc_end / dt))
    r_meas = abs(refl) / abs(inc)
    R, _ = scattering(prof, N_LEAD_C, N_TAIL_C, F0, dt, dy, b)
    r_model = abs(R)
    half = 0.20 * r_model + FS2_FLOOR
    return {
        "n_b": n_b, "band_mm": n_b * fx.DZ_FINE * 1e3,
        "builder_matches_declared_vector": builder_exact,
        "profile_cells_mm": (prof * 1e3).tolist(),
        "nz": int(len(prof)), "dt_s": dt,
        "R_model": r_model, "R_model_sq": r_model ** 2,
        "R_model_db": 20 * np.log10(r_model),
        "window": [r_model - half, r_model + half],
        "R_meas": r_meas, "R_meas_sq": r_meas ** 2,
        "R_meas_db": 20 * np.log10(max(r_meas, 1e-300)),
        "deviation": abs(r_meas - r_model),
        "f8_fired": bool(abs(r_meas - r_model) > half),
        "is_gate_row": n_b == 4,
        "gates_ns": {"t_r": t_r * 1e9, "t_band_last": t_band * 1e9,
                     "t_s": t_s * 1e9, "t_f": t_f * 1e9,
                     "gate_end": gate_end * 1e9, "gate_steps": n_gate,
                     "t_inc_end": t_inc_end * 1e9},
    }


def run_f8(widths: list[int]) -> dict:
    prof_b = b_profile()
    grid_b, trace_b = _run_probe(prof_b, N_STEPS)
    out = {"dt_b_s": float(grid_b.dt), "n_steps": N_STEPS, "K_SRC": K_SRC,
           "K_PRB": K_PRB, "coarse_cell_mm": DC * 1e3, "ramp_cell_mm": DR * 1e3,
           "fine_cell_mm": fx.DZ_FINE * 1e3, "rows": []}
    for n_b in widths:
        res = f8_arm(n_b, trace_b, grid_b)
        out["rows"].append(res)
        print(f"F8 n_b={n_b}: builder_exact={res['builder_matches_declared_vector']} "
              f"R_meas={res['R_meas']:.4e} ({res['R_meas_db']:.1f} dB) "
              f"R_model={res['R_model']:.4e} window=[{res['window'][0]:.4e}, "
              f"{res['window'][1]:.4e}] fired={res['f8_fired']} "
              f"gates(ns) t_r={res['gates_ns']['t_r']:.3f} "
              f"t_band={res['gates_ns']['t_band_last']:.3f} "
              f"gate_end={res['gates_ns']['gate_end']:.3f}", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="2,4,8,16")
    ap.add_argument("--out", default="validation/research/multiband_nu/results/w6_band_builder.json")
    ap.add_argument("--f7-only", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    results = {"rfx_file": rfx.__file__, "argv": sys.argv[1:],
               "git_sha": _git_sha(), "git_dirty": _git_dirty(),
               "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0))}
    results["f7"] = run_f7()
    if not args.f7_only:
        widths = [int(w) for w in args.widths.split(",") if w]
        results["f8"] = run_f8(widths)
    results["wallclock_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
