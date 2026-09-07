"""W8 -- G1 kernel ablation bench (NU cost-reduction lane).

Question. On the RTX 4090 the NU lane runs at 0.81x (300^3) / 0.90x (400^3)
of the uniform lane under the CPML-8 bench condition (plan note, "G1
baseline -- measured", VESSL run 369367259061). Under that condition BOTH
lanes run their slow paths (the uniform fused fast path is gated on
``not use_cpml``; ``rfx/simulation.py`` ``_fast_eligible``), so the gap is
made of: the NU code path itself (six 1-D ``inv_*`` broadcast multiplies in
both curls, per-step ``sigma_dt_2eps``/``ca``/``cb`` recomputation that XLA
may or may not hoist out of the ``lax.scan`` body, the separate
``apply_pec`` pass) and the graded-z memory access. This instrument
ATTRIBUTES the gap to those pieces by ablation; it changes nothing in
``rfx/``.

Method. ``scripts/diagnostics/gpu_throughput_bench.py``'s marginal-cost
differencing (``measure()`` copied verbatim in shape: time ``run(64)`` and
``run(1088)``, take cells*1024/(t_big - t_small); 3 windows, median and
spread), fp32, one soft source, no monitors. Fixtures:

* ``cpml8`` -- CPML 8 on all faces (the baseline condition), n = 300, 400.
* ``pec`` -- PEC-closed box, ``cpml_layers=0`` (attribution of the
  separate PEC pass), n = 300.

Arms. Each is a documented monkeypatch applied INSIDE this process on the
module attribute the step body binds (``rfx.nonuniform.update_e_nu`` etc.),
never a change to ``rfx/``:

    bare             uniform lane, uniform mesh (reference for the gap)
    nu-uniform       NU lane, uniform-valued dz_profile (NU code path, no
                     grading -- separates "NU code" from "graded access")
    nu-z             NU lane, 4:1 graded (the baseline arm)
    nu-z-hoist       nu-z + update_e_nu's sigma_dt_2eps/ca/cb computed ONCE
                     outside the scan body (jax.ensure_compile_time_eval on
                     the concrete material arrays; same expression, same
                     op order) -- bit-identity expected
    bare-hoist       the same hoist on the uniform update_e -- says whether
                     the hoist is an NU-specific or a common win
    nu-z-nopec       nu-z with apply_pec skipped (pec fixture only; physics
                     NOT identical -- attribution only, flagged)
    nu-z-scalar-inv  NU lane on the uniform-valued mesh with the six inv_*
                     broadcast vectors replaced by Python-float scalars of
                     the same float32 value (attribution of the broadcast
                     cost; must be bit-identical to nu-uniform there)

Bit-identity check (mandatory, before any timing): every patched arm is
run for 64 steps on a 96^3 box of its fixture with and without its patch
and every field (ex..hz) is compared with ``np.array_equal``. An arm whose
patch moves a bit is still timed but is flagged and can never become an
implementation candidate (the nopec arm is expected to differ).

Implementation rule (pre-declared in the plan note): an arm becomes an
implementation candidate only if it is bit-identical AND beats its
reference by more than 2x the larger of the two spreads AND by >= 3 %, on
BOTH 300^3 and 400^3. Sum check: the attributed pieces (graded access,
broadcast, NU-specific hoist) must sum to no more than the measured gap
plus the spread tolerance -- if they sum to more, the instrument is wrong.

Usage::

    python w8_nu_kernel_ablation.py            # full GPU matrix
    python w8_nu_kernel_ablation.py --smoke    # CPU: tiny n, every arm,
                                               # every bit-identity check
    python w8_nu_kernel_ablation.py --out /path/results.json

Writes ``w8_nu_kernel_ablation_results.json`` next to itself by default.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

import numpy as np

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------
# Pre-declared numbers (plan note, "G1 ablation -- pre-declared").
# Written before the first GPU run; never edited after.
# --------------------------------------------------------------------------
CANDIDATE_MIN_GAIN = 0.03      # arm must beat its reference by >= 3 %
CANDIDATE_SPREAD_MULT = 2.0    # ... and by more than 2x the larger spread
CANDIDATE_SIZES = (300, 400)   # ... on BOTH of these
FIXTURE_SIZES = {"cpml8": (300, 400), "pec": (300,)}
STEPS_SMALL, STEPS_BIG = 64, 1088
IDENTITY_N, IDENTITY_STEPS = 96, 64
# Preflight is a deterministic host-side check inside every run() call --
# part of the fixed cost the differencing subtracts -- skipped here only
# to keep its prints out of the log; recorded in the JSON meta.
RUN_KWARGS = dict(skip_preflight=True)

# arm -> (lane, mesh, patch name or None, reference arm for its patch,
#         identity expected?, fixtures it runs on)
ARMS = {
    "bare":            ("uniform", "uniform", None,          None,         None,  ("cpml8", "pec")),
    "nu-uniform":      ("nu",      "uniform", None,          None,         None,  ("cpml8", "pec")),
    "nu-z":            ("nu",      "graded",  None,          None,         None,  ("cpml8", "pec")),
    "nu-z-hoist":      ("nu",      "graded",  "hoist_nu",    "nu-z",       True,  ("cpml8",)),
    "bare-hoist":      ("uniform", "uniform", "hoist_u",     "bare",       True,  ("cpml8",)),
    "nu-z-nopec":      ("nu",      "graded",  "nopec",       "nu-z",       False, ("pec",)),
    "nu-z-scalar-inv": ("nu",      "uniform", "scalar_inv",  "nu-uniform", True,  ("cpml8",)),
}
ARM_ORDER = list(ARMS)


# --------------------------------------------------------------------------
# Patches. Each is a context manager that swaps a module attribute the
# step body looks up by name and restores it on exit. ``record`` collects
# trace-time facts (did the hoist really produce concrete arrays? was the
# uniform fused fast path used?) so the JSON says what actually ran.
# --------------------------------------------------------------------------

def _is_tracer(x) -> bool:
    return isinstance(x, jax.core.Tracer)


@contextlib.contextmanager
def _swap(module, name, replacement):
    orig = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield
    finally:
        setattr(module, name, orig)


def _coeffs_hoisted(materials, dt, record):
    """sigma_dt_2eps / ca / cb with the SAME expression and op order as
    ``update_e`` / ``update_e_nu`` (rfx/core/yee.py), evaluated at trace
    time on the concrete material arrays so the scan body closes over
    constants instead of recomputing them every step.

    Omnistaging stages every jnp op inside a trace, even on closed-over
    constants -- that is exactly why the kernels "recompute every step" --
    so the hoist needs ``jax.ensure_compile_time_eval()``. If the materials
    are tracers (an outer jit) it silently stays staged; ``record`` says
    which happened, so the arm cannot claim a hoist it did not get.
    """
    from rfx.core.yee import EPS_0
    with jax.ensure_compile_time_eval():
        eps = materials.eps_r * EPS_0
        sigma = materials.sigma
        sigma_dt_2eps = sigma * dt / (2.0 * eps)
        ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
        cb = (dt / eps) / (1.0 + sigma_dt_2eps)
    hoisted = not (_is_tracer(ca) or _is_tracer(cb))
    record.setdefault("hoisted", []).append(bool(hoisted))
    return ca, cb


def patch_hoist_nu(record):
    """nu-z-hoist: rfx.nonuniform.update_e_nu with hoisted coefficients."""
    import rfx.nonuniform as nu_mod
    from rfx.core.yee import curl_h_nu

    def update_e_nu_hoisted(state, materials, dt, inv_dx, inv_dy, inv_dz):
        _fdtype = state.ex.dtype
        _cdtype = jnp.promote_types(_fdtype, jnp.float32)
        hx = state.hx.astype(_cdtype)
        hy = state.hy.astype(_cdtype)
        hz = state.hz.astype(_cdtype)
        ca, cb = _coeffs_hoisted(materials, dt, record)
        curl_x, curl_y, curl_z = curl_h_nu(hx, hy, hz, inv_dx, inv_dy, inv_dz)
        ex = (ca * state.ex.astype(_cdtype) + cb * curl_x).astype(_fdtype)
        ey = (ca * state.ey.astype(_cdtype) + cb * curl_y).astype(_fdtype)
        ez = (ca * state.ez.astype(_cdtype) + cb * curl_z).astype(_fdtype)
        return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)

    return _swap(nu_mod, "update_e_nu", update_e_nu_hoisted)


def patch_hoist_uniform(record):
    """bare-hoist: rfx.simulation.update_e with hoisted coefficients
    (2nd-order real stencil only -- the bench never uses order 4 / Bloch)."""
    import rfx.simulation as sim_mod
    from rfx.core.yee import curl_h

    def update_e_hoisted(state, materials, dt, dx, periodic=(False, False, False),
                         stencil_order=2, bloch=None):
        if stencil_order != 2 or bloch is not None:
            raise ValueError("bare-hoist supports the order-2 real stencil only")
        _fdtype = state.ex.dtype
        _cdtype = jnp.promote_types(state.ex.dtype, jnp.float32)
        hx = state.hx.astype(_cdtype)
        hy = state.hy.astype(_cdtype)
        hz = state.hz.astype(_cdtype)
        ca, cb = _coeffs_hoisted(materials, dt, record)
        curl_x, curl_y, curl_z = curl_h(hx, hy, hz, dx, periodic, stencil_order, bloch)
        ex = (ca * state.ex.astype(_cdtype) + cb * curl_x).astype(_fdtype)
        ey = (ca * state.ey.astype(_cdtype) + cb * curl_y).astype(_fdtype)
        ez = (ca * state.ez.astype(_cdtype) + cb * curl_z).astype(_fdtype)
        return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)

    return _swap(sim_mod, "update_e", update_e_hoisted)


def patch_nopec(record):
    """nu-z-nopec: rfx.nonuniform.apply_pec becomes the identity. Physics
    NOT identical (tangential E at the faces is no longer zeroed) --
    attribution of the separate pass only."""
    import rfx.nonuniform as nu_mod

    def apply_pec_skipped(state, axes: str = "xyz"):
        record["pec_skipped"] = True
        return state

    return _swap(nu_mod, "apply_pec", apply_pec_skipped)


def _uniform_scalar(arr, name, record, drop_last=False):
    """The single float32 value of a uniform-valued inv_* array, as a Python
    float (weak-typed: ``f32_array * py_float`` multiplies by exactly this
    float32 value, so the product is the same bits as the broadcast)."""
    with jax.ensure_compile_time_eval():
        host = np.asarray(arr)
    body = host[:-1] if drop_last else host
    if body.size == 0 or not np.all(body == body[0]):
        raise ValueError(f"{name} is not uniform-valued: scalar-inv arm is "
                         "only defined on the uniform-valued mesh")
    record.setdefault("scalar_inv_values", {})[name] = float(body[0])
    return float(body[0])


def patch_scalar_inv(record):
    """nu-z-scalar-inv: update_h_nu / update_e_nu with the six 1-D inv_*
    broadcast vectors replaced by scalars of the same float32 value. On the
    uniform-valued mesh inv_d_e[k] = 2/(d+d) == 1/d exactly in IEEE (exact
    power-of-two scaling), and inv_d_h[N-1] = 0 only ever multiplies
    (0 - E_tan[N-1]) where E_tan is zeroed by apply_pec, so the physics and
    the bits are unchanged there. The check proves it."""
    import rfx.nonuniform as nu_mod
    from rfx.core.yee import MU_0, _shift_fwd, _shift_bwd

    def update_h_nu_scalar(state, materials, dt, inv_dx_h, inv_dy_h, inv_dz_h):
        sx = _uniform_scalar(inv_dx_h, "inv_dx_h", record, drop_last=True)
        sy = _uniform_scalar(inv_dy_h, "inv_dy_h", record, drop_last=True)
        sz = _uniform_scalar(inv_dz_h, "inv_dz_h", record, drop_last=True)
        _fdtype = state.ex.dtype
        _cdtype = jnp.promote_types(_fdtype, jnp.float32)
        ex = state.ex.astype(_cdtype)
        ey = state.ey.astype(_cdtype)
        ez = state.ez.astype(_cdtype)
        mu = materials.mu_r * MU_0
        curl_x = (_shift_fwd(ez, 1) - ez) * sy - (_shift_fwd(ey, 2) - ey) * sz
        curl_y = (_shift_fwd(ex, 2) - ex) * sz - (_shift_fwd(ez, 0) - ez) * sx
        curl_z = (_shift_fwd(ey, 0) - ey) * sx - (_shift_fwd(ex, 1) - ex) * sy
        hx = (state.hx.astype(_cdtype) - (dt / mu) * curl_x).astype(_fdtype)
        hy = (state.hy.astype(_cdtype) - (dt / mu) * curl_y).astype(_fdtype)
        hz = (state.hz.astype(_cdtype) - (dt / mu) * curl_z).astype(_fdtype)
        return state._replace(hx=hx, hy=hy, hz=hz)

    def update_e_nu_scalar(state, materials, dt, inv_dx, inv_dy, inv_dz):
        from rfx.core.yee import EPS_0
        sx = _uniform_scalar(inv_dx, "inv_dx", record)
        sy = _uniform_scalar(inv_dy, "inv_dy", record)
        sz = _uniform_scalar(inv_dz, "inv_dz", record)
        _fdtype = state.ex.dtype
        _cdtype = jnp.promote_types(_fdtype, jnp.float32)
        hx = state.hx.astype(_cdtype)
        hy = state.hy.astype(_cdtype)
        hz = state.hz.astype(_cdtype)
        eps = materials.eps_r * EPS_0
        sigma = materials.sigma
        sigma_dt_2eps = sigma * dt / (2.0 * eps)
        ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
        cb = (dt / eps) / (1.0 + sigma_dt_2eps)
        curl_x = (hz - _shift_bwd(hz, 1)) * sy - (hy - _shift_bwd(hy, 2)) * sz
        curl_y = (hx - _shift_bwd(hx, 2)) * sz - (hz - _shift_bwd(hz, 0)) * sx
        curl_z = (hy - _shift_bwd(hy, 0)) * sx - (hx - _shift_bwd(hx, 1)) * sy
        ex = (ca * state.ex.astype(_cdtype) + cb * curl_x).astype(_fdtype)
        ey = (ca * state.ey.astype(_cdtype) + cb * curl_y).astype(_fdtype)
        ez = (ca * state.ez.astype(_cdtype) + cb * curl_z).astype(_fdtype)
        return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)

    @contextlib.contextmanager
    def both():
        with _swap(nu_mod, "update_h_nu", update_h_nu_scalar), \
             _swap(nu_mod, "update_e_nu", update_e_nu_scalar):
            yield

    return both()


PATCHES = {
    "hoist_nu": patch_hoist_nu,
    "hoist_u": patch_hoist_uniform,
    "nopec": patch_nopec,
    "scalar_inv": patch_scalar_inv,
}


@contextlib.contextmanager
def _fast_path_probe(record):
    """Wrap rfx.simulation.update_he_fast so the JSON records whether the
    uniform fused fast path was traced (it is on GPU for the pec fixture:
    that arm then IS the fused target, not the slow path)."""
    import rfx.simulation as sim_mod
    orig = sim_mod.update_he_fast

    def probe(st, coeffs):
        record["fast_path_seen"] = True
        return orig(st, coeffs)

    with _swap(sim_mod, "update_he_fast", probe):
        yield


@contextlib.contextmanager
def arm_context(arm, record):
    patch = ARMS[arm][2]
    with _fast_path_probe(record):
        if patch is None:
            yield
        else:
            with PATCHES[patch](record):
                yield


# --------------------------------------------------------------------------
# Fixture (same cube, source and materials as gpu_throughput_bench.build)
# --------------------------------------------------------------------------

def run_fixture(n, lane, mesh, fixture, n_steps):
    """Build the bench cube for one (lane, mesh, fixture) and run it for
    ``n_steps``. Build and first solve live in one function on purpose:
    the patches must be in place when the Simulation compiles, and the
    example-fidelity contract classifies this script as
    ``builder_fused_with_solve``."""
    from rfx import Simulation, GaussianPulse
    L = n * 1e-3
    kw = {}
    if lane == "nu":
        if mesh == "graded":
            r = np.linspace(0.4, 1.6, n)          # 4:1 graded z, sum == L
            kw["dz_profile"] = r * (L / r.sum())
        else:
            kw["dz_profile"] = np.full(n, 1e-3)   # uniform-valued profile
    if fixture == "cpml8":
        bkw = dict(boundary="cpml", cpml_layers=8)
    elif fixture == "pec":
        bkw = dict(boundary="pec", cpml_layers=0)
    else:
        raise ValueError(fixture)
    sim = Simulation(freq_max=15e9, domain=(L, L, L), dx=1e-3, **bkw, **kw)
    sim.add_source((L / 2, L / 2, L / 2), "ez",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.7),
                   amplitude_kind="current")
    result = sim.run(n_steps=n_steps, **RUN_KWARGS)
    return sim, result


def _fields(result):
    st = result.state
    return {k: np.asarray(getattr(st, k)) for k in ("ex", "ey", "ez", "hx", "hy", "hz")}


# --------------------------------------------------------------------------
# Bit-identity check
# --------------------------------------------------------------------------

def bit_identity(arm, fixture, n, steps):
    """Run ``steps`` steps of the arm's configuration with and without its
    patch; np.array_equal on every field. Returns the record."""
    lane, mesh, patch, ref_arm, expected, _ = ARMS[arm]
    ref_lane, ref_mesh = ARMS[ref_arm][0], ARMS[ref_arm][1]
    assert (lane, mesh) == (ref_lane, ref_mesh), (arm, ref_arm)
    rec_ref: dict = {}
    with _fast_path_probe(rec_ref):
        _, r_ref = run_fixture(n, lane, mesh, fixture, steps)
    f_ref = _fields(r_ref)
    rec_arm: dict = {}
    with arm_context(arm, rec_arm):
        _, r_arm = run_fixture(n, lane, mesh, fixture, steps)
    f_arm = _fields(r_arm)
    per_field = {}
    for k in f_ref:
        same = bool(np.array_equal(f_ref[k], f_arm[k]))
        per_field[k] = dict(
            identical=same,
            max_abs_diff=float(np.max(np.abs(f_ref[k].astype(np.float64)
                                             - f_arm[k].astype(np.float64)))),
            max_abs_ref=float(np.max(np.abs(f_ref[k]))),
            n_diff=int(np.count_nonzero(f_ref[k] != f_arm[k])),
        )
    identical = all(v["identical"] for v in per_field.values())
    return dict(arm=arm, reference=ref_arm, fixture=fixture, n=n, steps=steps,
                identical=identical, expected_identical=expected,
                as_expected=(identical == expected),
                dtype=str(f_ref["ex"].dtype), per_field=per_field,
                patch_record=rec_arm)


# --------------------------------------------------------------------------
# Timing (gpu_throughput_bench.measure, with the arm's patch in place)
# --------------------------------------------------------------------------

def measure(arm, fixture, n, *, n1, n2, windows):
    lane, mesh, patch, _, _, _ = ARMS[arm]
    rec: dict = {}
    with arm_context(arm, rec):
        t0 = time.perf_counter()
        sim, r = run_fixture(n, lane, mesh, fixture, 8)
        t_compile = time.perf_counter() - t0
        cells = int(np.prod(np.asarray(r.state.ez).shape))
        jax.block_until_ready(sim.run(n_steps=n1, **RUN_KWARGS).state.ez)
        jax.block_until_ready(sim.run(n_steps=n2, **RUN_KWARGS).state.ez)
        meds = []
        for _ in range(windows):
            t0 = time.perf_counter()
            jax.block_until_ready(sim.run(n_steps=n1, **RUN_KWARGS).state.ez)
            t1 = time.perf_counter()
            jax.block_until_ready(sim.run(n_steps=n2, **RUN_KWARGS).state.ez)
            t2 = time.perf_counter()
            dt_steps = (t2 - t1) - (t1 - t0)
            if dt_steps <= 0:
                continue
            meds.append(cells * (n2 - n1) / dt_steps / 1e6)
        del sim, r
    if not meds:
        raise RuntimeError("differencing produced no positive window")
    return dict(arm=arm, lane=lane, mesh=mesh, fixture=fixture, n=n, cells=cells,
                steps=f"{n1}->{n2}", windows=len(meds),
                mc_per_s_median=float(np.median(meds)),
                mc_per_s_spread=float(np.max(meds) - np.min(meds)),
                mc_per_s_windows=[float(m) for m in meds],
                compile_s=round(t_compile, 2),
                fast_path_seen=bool(rec.get("fast_path_seen", False)),
                patch_record={k: v for k, v in rec.items() if k != "fast_path_seen"})


# --------------------------------------------------------------------------
# Pre-declared evaluation: candidate rule and the sum check
# --------------------------------------------------------------------------

def _row(rows, arm, fixture, n):
    for r in rows:
        if r.get("arm") == arm and r.get("fixture") == fixture and r.get("n") == n \
                and "mc_per_s_median" in r:
            return r
    return None


def evaluate(rows, identity, sizes=CANDIDATE_SIZES, fixture_sizes=None):
    fixture_sizes = fixture_sizes or FIXTURE_SIZES
    ident = {(d["arm"], d["fixture"]): d["identical"] for d in identity}
    verdicts = {}
    for arm, (lane, mesh, patch, ref_arm, expected, fixtures) in ARMS.items():
        if patch is None:
            continue
        per_n = {}
        for fx in fixtures:
            for n in fixture_sizes.get(fx, ()):
                a, b = _row(rows, arm, fx, n), _row(rows, ref_arm, fx, n)
                if a is None or b is None:
                    per_n[f"{fx}/{n}"] = dict(measured=False)
                    continue
                gain = a["mc_per_s_median"] / b["mc_per_s_median"] - 1.0
                delta = a["mc_per_s_median"] - b["mc_per_s_median"]
                spread = max(a["mc_per_s_spread"], b["mc_per_s_spread"])
                per_n[f"{fx}/{n}"] = dict(
                    measured=True, gain=gain, delta_mc_per_s=delta,
                    larger_spread=spread,
                    passes=(gain >= CANDIDATE_MIN_GAIN
                            and delta > CANDIDATE_SPREAD_MULT * spread))
        bit_ok = all(ident.get((arm, fx), False) for fx in fixtures)
        needed = [f"cpml8/{n}" for n in sizes]
        candidate = (bit_ok and expected is True
                     and all(per_n.get(k, {}).get("passes", False) for k in needed))
        verdicts[arm] = dict(reference=ref_arm, bit_identical=bit_ok,
                             per_size=per_n, implementation_candidate=candidate,
                             rule=(f"bit-identical AND gain >= {CANDIDATE_MIN_GAIN:.0%} "
                                   f"AND delta > {CANDIDATE_SPREAD_MULT:g}x larger spread, "
                                   f"on {needed}"))
    # Attribution in cost units c = 1/rate, normalised by c_bare.
    attribution = {}
    for fx, ns in fixture_sizes.items():
        for n in ns:
            def c(arm):
                r = _row(rows, arm, fx, n)
                return None if r is None else 1.0 / r["mc_per_s_median"]

            def s(arm):
                r = _row(rows, arm, fx, n)
                if r is None:
                    return 0.0
                m, sp = r["mc_per_s_median"], r["mc_per_s_spread"]
                return abs(1.0 / (m - sp / 2) - 1.0 / (m + sp / 2)) if m > sp / 2 else 1.0 / m
            cb, cz, cu = c("bare"), c("nu-z"), c("nu-uniform")
            if cb is None or cz is None:
                continue
            gap = cz / cb - 1.0
            pieces, tol = {}, 0.0
            if cu is not None:
                pieces["graded_access"] = (cz - cu) / cb
                pieces["nu_code_path"] = (cu - cb) / cb
                tol += (s("nu-z") + s("nu-uniform")) / cb
            csi = c("nu-z-scalar-inv")
            if cu is not None and csi is not None:
                pieces["broadcast_inv"] = (cu - csi) / cb
                tol += (s("nu-uniform") + s("nu-z-scalar-inv")) / cb
            czh, cbh = c("nu-z-hoist"), c("bare-hoist")
            if czh is not None:
                pieces["hoist_total"] = (cz - czh) / cb
                tol += (s("nu-z") + s("nu-z-hoist")) / cb
                if cbh is not None:
                    pieces["hoist_common"] = (cb - cbh) / cb
                    pieces["hoist_nu_specific"] = ((cz - czh) - (cb - cbh)) / cb
                    tol += (s("bare") + s("bare-hoist")) / cb
            cnp = c("nu-z-nopec")
            if cnp is not None:
                pieces["pec_pass"] = (cz - cnp) / cb
                tol += (s("nu-z") + s("nu-z-nopec")) / cb
            # Sum check: the disjoint pieces that partition the gap.
            summed = sum(max(pieces.get(k, 0.0), 0.0) for k in
                         ("graded_access", "broadcast_inv", "hoist_nu_specific", "pec_pass"))
            attribution[f"{fx}/{n}"] = dict(
                ratio_nuz_over_bare=cb / cz, gap_cost=gap, pieces=pieces,
                sum_of_pieces=summed, spread_tolerance=tol,
                sum_check_ok=(summed <= gap + tol),
                sum_check_rule="graded_access + broadcast_inv + hoist_nu_specific "
                               "+ pec_pass (each clipped at 0) <= gap + spread tolerance; "
                               "if not, the instrument is wrong")
    return dict(candidates=verdicts, attribution=attribution)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def _git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    for cwd in (here, os.getcwd()):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            continue
    for p in (os.path.join(here, ".staged-commit"),
              os.path.join(here, "..", ".staged-commit")):
        if os.path.exists(p):
            return open(p).read().strip()
    return "unknown"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="CPU smoke: n=32 (identity box 32^3), 8->136 steps, every arm and check")
    ap.add_argument("--out", default=None)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--only-fixture", choices=list(FIXTURE_SIZES), default=None)
    args = ap.parse_args(argv)

    if args.smoke:
        sizes = {"cpml8": (32,), "pec": (32,)}
        n1, n2, windows = 8, 136, args.windows
        id_n, id_steps = 32, IDENTITY_STEPS
    else:
        sizes = dict(FIXTURE_SIZES)
        n1, n2, windows = STEPS_SMALL, STEPS_BIG, args.windows
        id_n, id_steps = IDENTITY_N, IDENTITY_STEPS
    if args.only_fixture:
        sizes = {args.only_fixture: sizes[args.only_fixture]}

    meta = dict(instrument="w8_nu_kernel_ablation", git_sha=_git_sha(),
                smoke=args.smoke, argv=sys.argv[1:], jax_version=jax.__version__,
                device=str(jax.devices()[0]), backend=jax.default_backend(),
                steps=f"{n1}->{n2}", windows=windows,
                identity=dict(n=id_n, steps=id_steps), run_kwargs=RUN_KWARGS,
                started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(json.dumps(meta), flush=True)

    # 1. Bit-identity checks, all before any timing.
    identity = []
    print("=== bit-identity checks ===", flush=True)
    for arm in ARM_ORDER:
        lane, mesh, patch, ref_arm, expected, fixtures = ARMS[arm]
        if patch is None:
            continue
        for fx in fixtures:
            if fx not in sizes:
                continue
            rec = bit_identity(arm, fx, id_n, id_steps)
            identity.append(rec)
            worst = max(v["max_abs_diff"] for v in rec["per_field"].values())
            flag = "OK" if rec["as_expected"] else "UNEXPECTED"
            print(f"  {arm:16s} vs {ref_arm:10s} [{fx}] {id_n}^3 x {id_steps}: "
                  f"identical={rec['identical']} expected={expected} -> {flag} "
                  f"(max|diff| {worst:.3e}; patch {rec['patch_record']})", flush=True)

    # 2. Timing.
    rows = []
    print("=== timing ===", flush=True)
    for fx, ns in sizes.items():
        for n in ns:
            for arm in ARM_ORDER:
                if fx not in ARMS[arm][5]:
                    continue
                try:
                    row = measure(arm, fx, n, n1=n1, n2=n2, windows=windows)
                    print(f"  [{fx}] {n}^3 {arm:16s}: {row['mc_per_s_median']:8.1f} "
                          f"Mcells/s (spread {row['mc_per_s_spread']:.1f}, "
                          f"fast_path={row['fast_path_seen']}, "
                          f"patch={row['patch_record']})", flush=True)
                except Exception as e:  # OOM -- say so, keep going
                    msg = f"{type(e).__name__}: {str(e)[:160]}"
                    print(f"  [{fx}] {n}^3 {arm:16s}: SKIPPED ({msg})", flush=True)
                    row = dict(arm=arm, fixture=fx, n=n, cells=n ** 3, skipped=msg)
                rows.append(row)

    evaluation = evaluate(rows, identity, fixture_sizes=sizes)
    out = dict(meta=meta, identity=identity, rows=rows, evaluation=evaluation,
               finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "w8_nu_kernel_ablation_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("=== evaluation ===", flush=True)
    print(json.dumps(evaluation, indent=1), flush=True)
    print(f"wrote {path}", flush=True)
    return out


if __name__ == "__main__":
    main()
