#!/usr/bin/env python3
"""WR-90 reference-free S11 and source-purity oracle matrix.

Diagnostic-only harness for separating current two-run reference/DFT/CPML
calibration artifacts from source impurity and internal short physics.  This
script intentionally avoids production source/extractor changes: it builds
script-local rows, emits machine-readable JSONL evidence, and leaves strict
closure/#13/#17 untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rfx.api import Simulation  # noqa: E402
from rfx.geometry.csg import Box  # noqa: E402
from rfx.simulation import run as run_simulation  # noqa: E402
from rfx.sources.waveguide_port import (  # noqa: E402
    WaveguidePortConfig,
    _compute_beta,
    _rect_dft,
    waveguide_plane_positions,
)

DOMAIN = (0.12, 0.04, 0.02)
PORT_LEFT_X = 0.01
PORT_RIGHT_X = 0.09
PEC_SHORT_X = 0.085
PEC_THICKNESS = 0.002
DEFAULT_CPML_LAYERS = 10
DEFAULT_NUM_PERIODS = 40.0
DEFAULT_FREQS_HZ = np.linspace(5.0e9, 7.0e9, 6)

Status = Literal["ok", "control", "skipped", "error"]


@dataclass(frozen=True)
class GammaFitResult:
    """Least-squares two-wave fit result for one or more frequencies."""

    a_plus: np.ndarray
    a_minus: np.ndarray
    gamma: np.ndarray
    residual_norm: np.ndarray
    condition: np.ndarray
    rank: np.ndarray

    @property
    def gamma_mag(self) -> np.ndarray:
        return np.abs(self.gamma)

    @property
    def gamma_phase_deg(self) -> np.ndarray:
        return np.rad2deg(np.angle(self.gamma))


@dataclass(frozen=True)
class OracleRow:
    """One machine-readable diagnostic row."""

    case: str
    method: str
    status: Status = "ok"
    metrics: dict[str, Any] = field(default_factory=dict)
    verdict_hint: str = "diagnostic_only"
    skip_reason: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "case": self.case,
            "method": self.method,
            "status": self.status,
            "verdict_hint": self.verdict_hint,
        }
        row.update(_json_safe(self.metrics))
        if self.skip_reason:
            row["skip_reason"] = self.skip_reason
        return row


@dataclass(frozen=True)
class OracleCaseConfig:
    """Physical WR-90 oracle case settings."""

    freqs_hz: np.ndarray = field(default_factory=lambda: DEFAULT_FREQS_HZ.copy())
    cpml_layers: int = DEFAULT_CPML_LAYERS
    num_periods: float = DEFAULT_NUM_PERIODS
    pec_short_x: float | None = PEC_SHORT_X
    monitor_x_m: tuple[float, ...] = (0.030, 0.045, 0.060)
    source_x_m: float = PORT_LEFT_X
    short_type: str = "internal_mask"
    waveform: str = "modulated_gaussian"
    mode_profile: str = "discrete"
    dx: float | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _freqs_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.center_freq_hz is not None:
        return np.asarray([float(args.center_freq_hz)], dtype=float)
    if args.n_freqs <= 1:
        return np.asarray([float(args.freq_min_hz)], dtype=float)
    return np.linspace(float(args.freq_min_hz), float(args.freq_max_hz), int(args.n_freqs))


def fit_two_wave_line(
    monitor_x_m: np.ndarray | Iterable[float],
    modal_voltage_dft: np.ndarray,
    beta: np.ndarray | complex | float,
) -> GammaFitResult:
    """Fit ``u(x)=a+ exp(-jβx)+a- exp(+jβx)`` with 3+ monitor planes.

    ``modal_voltage_dft`` may be shape ``(n_planes,)`` for one frequency or
    ``(n_planes, n_freqs)`` for a band.  ``beta`` is scalar or ``(n_freqs,)``.
    The returned ``gamma`` is ``a_minus/a_plus`` at the global coordinate
    origin used by ``monitor_x_m``; its magnitude is independent of origin.
    """

    x = np.asarray(list(monitor_x_m), dtype=float)
    u = np.asarray(modal_voltage_dft, dtype=np.complex128)
    if u.ndim == 1:
        u = u[:, None]
    if u.ndim != 2:
        raise ValueError("modal_voltage_dft must be 1-D or 2-D")
    if u.shape[0] != x.size:
        raise ValueError(
            f"monitor_x_m has {x.size} planes but modal_voltage_dft has {u.shape[0]} rows"
        )
    if x.size < 2:
        raise ValueError("at least two monitor planes are required")

    beta_arr = np.asarray(beta, dtype=np.complex128)
    if beta_arr.ndim == 0:
        beta_arr = np.full((u.shape[1],), beta_arr.item(), dtype=np.complex128)
    if beta_arr.shape != (u.shape[1],):
        raise ValueError(f"beta shape {beta_arr.shape} does not match n_freqs={u.shape[1]}")

    a_plus = np.empty(u.shape[1], dtype=np.complex128)
    a_minus = np.empty(u.shape[1], dtype=np.complex128)
    gamma = np.empty(u.shape[1], dtype=np.complex128)
    residual = np.empty(u.shape[1], dtype=np.float64)
    cond = np.empty(u.shape[1], dtype=np.float64)
    rank = np.empty(u.shape[1], dtype=np.int64)

    for idx, beta_i in enumerate(beta_arr):
        design = np.column_stack((np.exp(-1j * beta_i * x), np.exp(+1j * beta_i * x)))
        solution, residuals, rank_i, singular_values = np.linalg.lstsq(design, u[:, idx], rcond=None)
        a_plus[idx], a_minus[idx] = solution
        gamma[idx] = a_minus[idx] / a_plus[idx] if abs(a_plus[idx]) > 1e-30 else np.nan + 1j * np.nan
        if residuals.size:
            residual[idx] = float(np.sqrt(residuals[0]) / max(np.linalg.norm(u[:, idx]), 1e-30))
        else:
            residual[idx] = float(np.linalg.norm(design @ solution - u[:, idx]) / max(np.linalg.norm(u[:, idx]), 1e-30))
        cond[idx] = float(np.inf if singular_values[-1] == 0 else singular_values[0] / singular_values[-1])
        rank[idx] = int(rank_i)

    return GammaFitResult(a_plus, a_minus, gamma, residual, cond, rank)


def solve_ref_free_gamma(
    monitor_x_m: np.ndarray | Iterable[float],
    modal_voltage_dft: np.ndarray,
    beta: np.ndarray | complex | float,
) -> GammaFitResult:
    """Public wrapper used by tests and the physical oracle rows."""

    return fit_two_wave_line(monitor_x_m, modal_voltage_dft, beta)


def integer_cycle_lockin(
    time_series: np.ndarray,
    freq_hz: float,
    dt: float,
    start_index: int,
    n_cycles: int,
) -> complex:
    """Return normalized complex phasor from an integer-period lock-in.

    For ``A*cos(2πft+φ)`` sampled over an integer number of periods, the
    return value is approximately ``A*exp(+jφ)``.  The helper is deliberately
    script-local; physical CW sources are not productionized in this PR.
    """

    if freq_hz <= 0 or dt <= 0:
        raise ValueError("freq_hz and dt must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if n_cycles <= 0:
        raise ValueError("n_cycles must be positive")

    arr = np.asarray(time_series, dtype=np.float64)
    samples_per_period = int(round(1.0 / (float(freq_hz) * float(dt))))
    if samples_per_period <= 0:
        raise ValueError("freq_hz*dt is too large to resolve one period")
    n_samples = int(samples_per_period * n_cycles)
    end = int(start_index) + n_samples
    if end > arr.size:
        raise ValueError("requested lock-in window exceeds time_series length")
    n = np.arange(start_index, end, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * float(freq_hz) * n * float(dt))
    return complex(2.0 * np.mean(arr[start_index:end] * phase))


def _build_wr90_two_port_sim(
    freqs_hz: np.ndarray,
    *,
    cpml_layers: int = DEFAULT_CPML_LAYERS,
    pec_short_x: float | None = PEC_SHORT_X,
    waveform: str = "modulated_gaussian",
    dx: float | None = None,
) -> Simulation:
    freqs = np.asarray(freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    sim_kwargs: dict[str, Any] = {
        "freq_max": max(float(freqs[-1]), f0),
        "domain": DOMAIN,
        "boundary": "cpml",
        "cpml_layers": int(cpml_layers),
    }
    if dx is not None:
        sim_kwargs["dx"] = float(dx)
    sim = Simulation(**sim_kwargs)
    if pec_short_x is not None:
        sim.add(
            Box((pec_short_x, 0.0, 0.0), (pec_short_x + PEC_THICKNESS, DOMAIN[1], DOMAIN[2])),
            material="pec",
        )
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        mode_profile="discrete",
    )
    sim.add_waveguide_port(PORT_LEFT_X, direction="+x", name="left", **common)
    sim.add_waveguide_port(PORT_RIGHT_X, direction="-x", name="right", **common)
    return sim


def _add_passive_monitor_ports(
    sim: Simulation,
    freqs: np.ndarray,
    monitor_x_m: Iterable[float],
    *,
    ref_offset: int = 3,
    probe_offset: int = 4,
    waveform: str = "modulated_gaussian",
    mode_profile: str = "discrete",
) -> None:
    grid = sim._build_grid(extra_waveguide_axes="x")
    dx = float(grid.dx)
    f0 = float(np.asarray(freqs, dtype=float).mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=waveform,
        mode_profile=mode_profile,
        amplitude=0.0,
        ref_offset=ref_offset,
        probe_offset=probe_offset,
    )
    for idx, x_monitor in enumerate(monitor_x_m):
        # ``reference_plane`` is only an extraction/reporting override in the
        # production S-matrix path, so for passive monitors we place the
        # zero-amplitude source plane upstream such that cfg.ref_x samples the
        # requested monitor plane after grid snapping.
        x_source = float(x_monitor) - ref_offset * dx
        sim.add_waveguide_port(
            x_source,
            direction="+x",
            name=f"monitor_{idx}",
            **common,
        )


def build_wr90_oracle_case(case_config: OracleCaseConfig) -> dict[str, Any]:
    """Build a one-run active-source plus passive-monitor oracle case."""

    freqs = np.asarray(case_config.freqs_hz, dtype=float)
    f0 = float(freqs.mean())
    bandwidth = max(0.2, min(0.8, (float(freqs[-1]) - float(freqs[0])) / max(f0, 1.0)))
    sim_kwargs: dict[str, Any] = {
        "freq_max": max(float(freqs[-1]), f0),
        "domain": DOMAIN,
        "boundary": "cpml",
        "cpml_layers": int(case_config.cpml_layers),
    }
    if case_config.dx is not None:
        sim_kwargs["dx"] = float(case_config.dx)
    sim = Simulation(**sim_kwargs)
    if case_config.pec_short_x is not None:
        sim.add(
            Box(
                (case_config.pec_short_x, 0.0, 0.0),
                (case_config.pec_short_x + PEC_THICKNESS, DOMAIN[1], DOMAIN[2]),
            ),
            material="pec",
        )

    active_common = dict(
        mode=(1, 0),
        mode_type="TE",
        freqs=jnp.asarray(freqs),
        f0=f0,
        bandwidth=bandwidth,
        waveform=case_config.waveform,
        mode_profile=case_config.mode_profile,
    )
    sim.add_waveguide_port(case_config.source_x_m, direction="+x", name="active_source", **active_common)
    _add_passive_monitor_ports(
        sim,
        freqs,
        case_config.monitor_x_m,
        waveform=case_config.waveform,
        mode_profile=case_config.mode_profile,
    )

    entries = list(sim._waveguide_ports)
    grid = sim._build_grid()
    base_materials, debye_spec, lorentz_spec, pec_mask_wg, _, _ = sim._assemble_materials(grid)
    if pec_mask_wg is not None:
        # Match the current waveguide S-matrix compatibility path so the
        # oracle compares against the same internal-mask representation.
        base_materials = base_materials._replace(sigma=jnp.where(pec_mask_wg, 1e10, base_materials.sigma))
    _, debye, lorentz = sim._init_dispersion(base_materials, grid.dt, debye_spec, lorentz_spec)
    n_steps = grid.num_timesteps(num_periods=case_config.num_periods)

    cfgs: list[WaveguidePortConfig] = [sim._build_waveguide_port_config(entry, grid, jnp.asarray(freqs), n_steps) for entry in entries]
    # Ensure all passive monitors are really passive even if future API default
    # changes alter amplitude handling.
    cfgs = [cfg if idx == 0 else cfg._replace(src_amp=0.0) for idx, cfg in enumerate(cfgs)]

    return {
        "grid": grid,
        "materials": base_materials,
        "debye": debye,
        "lorentz": lorentz,
        "cfgs": cfgs,
        "n_steps": n_steps,
        "common_run_kw": dict(
            boundary="cpml",
            cpml_axes=grid.cpml_axes,
            pec_axes="".join(axis for axis in "xyz" if axis not in grid.cpml_axes),
            periodic=None,
        ),
    }


def _run_waveguide_case(case: dict[str, Any]) -> list[WaveguidePortConfig]:
    result = run_simulation(
        case["grid"],
        case["materials"],
        case["n_steps"],
        debye=case["debye"],
        lorentz=case["lorentz"],
        waveguide_ports=case["cfgs"],
        **case["common_run_kw"],
    )
    return list(result.waveguide_ports or ())


def _voltage_spectrum_at_reference(cfg: WaveguidePortConfig) -> np.ndarray:
    return np.asarray(_rect_dft(cfg.v_ref_t, cfg.freqs, cfg.dt, cfg.n_steps_recorded), dtype=np.complex128)


def run_reference_free_case(case_config: OracleCaseConfig, *, case_name: str) -> OracleRow:
    started = time.perf_counter()
    try:
        case = build_wr90_oracle_case(case_config)
        final_cfgs = _run_waveguide_case(case)
        if len(final_cfgs) < 4:
            raise RuntimeError(f"expected active source + >=3 passive monitors, got {len(final_cfgs)} configs")
        monitor_cfgs = final_cfgs[1:]
        monitor_x = np.asarray([waveguide_plane_positions(cfg)["reference"] for cfg in monitor_cfgs], dtype=float)
        spectra = np.vstack([_voltage_spectrum_at_reference(cfg) for cfg in monitor_cfgs])
        beta = np.asarray(
            _compute_beta(
                monitor_cfgs[0].freqs,
                monitor_cfgs[0].f_cutoff,
                dt=monitor_cfgs[0].dt,
                dx=monitor_cfgs[0].dx,
            ),
            dtype=np.complex128,
        )
        fit = solve_ref_free_gamma(monitor_x, spectra, beta)
        freqs = np.asarray(monitor_cfgs[0].freqs, dtype=float)
        s11_mag = fit.gamma_mag
        metrics = {
            "freq_hz": float(freqs[len(freqs) // 2]),
            "freqs_hz": freqs,
            "s11_mag": float(s11_mag[len(s11_mag) // 2]),
            "s11_phase_deg": float(fit.gamma_phase_deg[len(s11_mag) // 2]),
            "mean_abs_s11": float(np.nanmean(s11_mag)),
            "min_abs_s11": float(np.nanmin(s11_mag)),
            "max_abs_s11": float(np.nanmax(s11_mag)),
            "mag_error_pct": float(100.0 * abs(np.nanmean(s11_mag) - 1.0)),
            "fit_residual": float(np.nanmean(fit.residual_norm)),
            "fit_cond": float(np.nanmax(fit.condition)),
            "fit_rank_min": int(np.nanmin(fit.rank)),
            "source": "analytic_te10_current",
            "short_type": case_config.short_type,
            "cpml_layers": int(case_config.cpml_layers),
            "num_periods": float(case_config.num_periods),
            "monitor_backend": "passive_waveguide_ref_voltage",
            "monitor_distances_m": monitor_x - float(case_config.source_x_m),
            "monitor_x_m": monitor_x,
            "requested_monitor_x_m": np.asarray(case_config.monitor_x_m, dtype=float),
            "source_short_distance_m": None if case_config.pec_short_x is None else float(case_config.pec_short_x - case_config.source_x_m),
            "monitor_short_distances_m": None if case_config.pec_short_x is None else float(case_config.pec_short_x) - monitor_x,
            "beta_type": "yee_discrete",
            "dft_window": "rect_full_record",
            "dft_type": "post_scan_rect_dft",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(case_name, "ref_free_multiplane", "ok", metrics, "A_D_or_C3_if_ref_free_deficit_persists")
    except Exception as exc:
        return OracleRow(
            case_name,
            "ref_free_multiplane",
            "error",
            {"elapsed_s": float(time.perf_counter() - started)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def run_current_2run_baseline(freqs_hz: np.ndarray, *, cpml_layers: int, num_periods: float, dx: float | None = None) -> OracleRow:
    started = time.perf_counter()
    try:
        sim = _build_wr90_two_port_sim(freqs_hz, cpml_layers=cpml_layers, pec_short_x=PEC_SHORT_X, dx=dx)
        result = sim.compute_waveguide_s_matrix(num_periods=num_periods, normalize=True)
        s = np.asarray(result.s_params, dtype=np.complex128)
        port_idx = {name: idx for idx, name in enumerate(result.port_names)}
        s11 = s[port_idx["left"], port_idx["left"], :]
        mag = np.abs(s11)
        freqs = np.asarray(result.freqs, dtype=float)
        metrics = {
            "freq_hz": float(freqs[len(freqs) // 2]),
            "freqs_hz": freqs,
            "s11_mag": float(mag[len(mag) // 2]),
            "s11_phase_deg": float(np.rad2deg(np.angle(s11[len(s11) // 2]))),
            "mean_abs_s11": float(np.mean(mag)),
            "min_abs_s11": float(np.min(mag)),
            "max_abs_s11": float(np.max(mag)),
            "mag_error_pct": float(100.0 * abs(np.mean(mag) - 1.0)),
            "source": "analytic_te10_current",
            "short_type": "internal_mask",
            "cpml_layers": int(cpml_layers),
            "num_periods": float(num_periods),
            "monitor_backend": "production_two_run_waveguide_s_matrix",
            "beta_type": "production_current",
            "dft_window": "rect_full_record",
            "dft_type": "current_2run_post_scan_rect_dft",
            "elapsed_s": float(time.perf_counter() - started),
        }
        return OracleRow(
            "baseline_current_2run_internal_mask_current_cpml",
            "current_2run",
            "ok",
            metrics,
            "B_or_C_if_ref_free_good",
        )
    except Exception as exc:
        return OracleRow(
            "baseline_current_2run_internal_mask_current_cpml",
            "current_2run",
            "error",
            {"elapsed_s": float(time.perf_counter() - started), "cpml_layers": int(cpml_layers)},
            "diagnostic_failed_no_closure_claim",
            f"{type(exc).__name__}: {exc}",
        )


def synthetic_least_squares_control() -> OracleRow:
    freqs = np.asarray([5.0e9, 6.0e9, 7.0e9])
    beta = np.asarray([95.0, 125.0, 150.0])
    x = np.asarray([0.024, 0.036, 0.049, 0.063])
    a_plus = np.asarray([1.0 + 0.0j, 0.8 + 0.1j, 1.1 - 0.05j])
    gamma_true = np.asarray([0.96 * np.exp(1j * 0.2), 0.98 * np.exp(-1j * 0.4), 1.02 * np.exp(1j * 0.1)])
    a_minus = gamma_true * a_plus
    samples = np.column_stack([
        a_plus[i] * np.exp(-1j * beta[i] * x) + a_minus[i] * np.exp(+1j * beta[i] * x)
        for i in range(freqs.size)
    ])
    fit = solve_ref_free_gamma(x, samples, beta)
    err = np.max(np.abs(fit.gamma - gamma_true))
    return OracleRow(
        "synthetic_ref_free_least_squares_control",
        "ref_free_multiplane_control",
        "control",
        {
            "freqs_hz": freqs,
            "gamma_error_max": float(err),
            "fit_residual_max": float(np.max(fit.residual_norm)),
            "fit_cond_max": float(np.max(fit.condition)),
            "monitor_backend": "synthetic_two_wave_line",
            "beta_type": "synthetic_discrete_beta",
            "dft_window": "not_applicable_synthetic",
            "dft_type": "not_applicable_synthetic",
        },
        "control_only_no_physical_closure",
    )


def cw_lockin_control_row() -> OracleRow:
    freq = 10.0e9
    samples_per_period = 40
    dt = 1.0 / (freq * samples_per_period)
    phase = 0.37
    amp = 1.25
    n = np.arange(samples_per_period * 12, dtype=float)
    signal = amp * np.cos(2.0 * np.pi * freq * n * dt + phase)
    phasor = integer_cycle_lockin(signal, freq, dt, start_index=samples_per_period * 2, n_cycles=8)
    return OracleRow(
        "cw_lockin_synthetic_control",
        "cw_integer_period_lockin_control",
        "control",
        {
            "freq_hz": freq,
            "target_amplitude": amp,
            "target_phase_rad": phase,
            "recovered_amplitude": float(abs(phasor)),
            "recovered_phase_rad": float(np.angle(phasor)),
            "amplitude_error": float(abs(abs(phasor) - amp)),
            "phase_error_rad": float(abs(np.angle(phasor * np.exp(-1j * phase)))),
            "warmup_cycles": 2,
            "lockin_cycles": 8,
            "dft_window": "integer_period_lockin",
            "dft_type": "cw_lockin_normalized_phasor",
        },
        "control_only_cw_helper_ready",
    )


def run_source_purity_empty_line_sweep(freqs_hz: np.ndarray, *, cpml_layers: int, num_periods: float, dx: float | None = None) -> OracleRow:
    monitors = (0.025, 0.040, 0.055, 0.070)
    cfg = OracleCaseConfig(
        freqs_hz=np.asarray(freqs_hz, dtype=float),
        cpml_layers=int(cpml_layers),
        num_periods=float(num_periods),
        pec_short_x=None,
        monitor_x_m=monitors,
        short_type="empty_guide",
        dx=dx,
    )
    row = run_reference_free_case(cfg, case_name="source_purity_empty_line_sweep")
    if row.status != "ok":
        return row
    metrics = dict(row.metrics)
    # For an empty guide the fitted backward/forward ratio is a practical
    # line-purity proxy; phase slope/absolute amplitude remain diagnostic.
    metrics["source_purity_metric"] = "empty_guide_ref_free_backward_forward_ratio"
    metrics["gamma_interpretation"] = "empty_guide backward/forward ratio, not PEC-short S11"
    metrics["residual_field_norm"] = metrics.get("fit_residual")
    metrics["poynting_flux"] = None
    return OracleRow(row.case, "source_purity_line_sweep", row.status, metrics, "A_if_distance_dependent_impurity_seen")


def skipped_row(case: str, method: str, reason: str, *, metrics: dict[str, Any] | None = None) -> OracleRow:
    return OracleRow(case, method, "skipped", metrics or {}, "explicit_skip_no_closure_claim", reason)


def emit_jsonl_rows(rows: Iterable[OracleRow], path_or_stdout: str | Path | None = None) -> str | None:
    lines = [json.dumps(row.to_jsonable(), sort_keys=True, allow_nan=False) for row in rows]
    payload = "\n".join(lines)
    if path_or_stdout is None or str(path_or_stdout) == "-":
        print(payload)
        return None
    path = Path(path_or_stdout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
    return str(path)


def format_human_table(rows: Iterable[OracleRow]) -> str:
    out = ["case | method | status | mean|Γ| | min|Γ| | fit_residual | note", "--- | --- | --- | ---: | ---: | ---: | ---"]
    for row in rows:
        data = row.to_jsonable()
        mean = data.get("mean_abs_s11")
        min_mag = data.get("min_abs_s11")
        fit = data.get("fit_residual")
        def fmt(value: Any) -> str:
            return "" if value is None else (f"{value:.6g}" if isinstance(value, (int, float)) else str(value))
        note = data.get("skip_reason") or data.get("verdict_hint", "")
        out.append(f"{row.case} | {row.method} | {row.status} | {fmt(mean)} | {fmt(min_mag)} | {fmt(fit)} | {note}")
    return "\n".join(out)


def run_matrix(args: argparse.Namespace) -> list[OracleRow]:
    freqs = _freqs_from_args(args)
    rows: list[OracleRow] = []
    rows.append(synthetic_least_squares_control())
    rows.append(cw_lockin_control_row())

    if args.synthetic_only:
        return rows

    rows.append(run_current_2run_baseline(freqs, cpml_layers=args.cpml_layers, num_periods=args.num_periods, dx=args.dx))
    cfg_3 = OracleCaseConfig(
        freqs_hz=freqs,
        cpml_layers=args.cpml_layers,
        num_periods=args.num_periods,
        monitor_x_m=tuple(args.monitor_x_m),
        dx=args.dx,
    )
    rows.append(run_reference_free_case(cfg_3, case_name="ref_free_3plane_internal_mask_current_cpml"))

    if args.full:
        cfg_5 = OracleCaseConfig(
            freqs_hz=freqs,
            cpml_layers=args.cpml_layers,
            num_periods=args.num_periods,
            monitor_x_m=(0.026, 0.036, 0.046, 0.056, 0.066),
            dx=args.dx,
        )
        rows.append(run_reference_free_case(cfg_5, case_name="ref_free_5plane_internal_mask_current_cpml"))
        rows.append(run_source_purity_empty_line_sweep(freqs, cpml_layers=args.cpml_layers, num_periods=args.num_periods, dx=args.dx))
        rows.append(
            skipped_row(
                "cw_lockin_ref_free_internal_mask_current_cpml",
                "cw_integer_period_ref_free",
                "physical CW waveguide source tables are intentionally not productionized in this diagnostic PR; helper is covered by cw_lockin_synthetic_control",
                metrics={"warmup_cycles": args.cw_warmup_cycles, "lockin_cycles": args.cw_lockin_cycles},
            )
        )
        rows.append(
            skipped_row(
                "source_short_distance_sweep",
                "ref_free_distance_sweep",
                "expensive multi-run sweep deferred behind the first quick oracle evidence; enable in follow-up once monitor backend is accepted",
            )
        )
        rows.append(
            skipped_row(
                "monitor_short_distance_sweep",
                "ref_free_distance_sweep",
                "expensive multi-run sweep deferred behind the first quick oracle evidence; current row records monitor_short_distances_m for the 3/5-plane cases",
            )
        )
        rows.append(
            skipped_row(
                "pml_sweep_current_2run_layers_10_20_40",
                "current_2run_cpml_sweep",
                "CPML 10/20/40 production two-run sweep is intentionally skipped by default to keep --full bounded; row is machine-readable for follow-up execution",
                metrics={"requested_cpml_layers": [10, 20, 40]},
            )
        )
        rows.append(
            skipped_row(
                "pml_sweep_ref_free_layers_10_20_40",
                "ref_free_cpml_sweep",
                "CPML 10/20/40 reference-free sweep is intentionally skipped by default to keep --full bounded; row is machine-readable for follow-up execution",
                metrics={"requested_cpml_layers": [10, 20, 40]},
            )
        )
        rows.append(
            skipped_row(
                "face_short_ref_free_no_cpml_or_irrelevant",
                "boundary_face_short_ref_free",
                "equivalent boundary-face short geometry cannot be constructed without production/API geometry changes in this PR",
                metrics={"comparison_quality": "qualitative_skipped"},
            )
        )
        rows.append(
            skipped_row(
                "internal_mask_short_ref_free_same_geometry",
                "internal_mask_short_ref_free",
                "covered by ref_free_3plane/ref_free_5plane internal-mask rows; boundary-face pair skipped qualitatively",
                metrics={"comparison_quality": "qualitative_skipped"},
            )
        )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", action="store_true", help="Run the WR-90 oracle matrix.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run MVP rows only (default).")
    mode.add_argument("--full", action="store_true", help="Attempt Matrix A-F, emitting skips for bounded/deferred rows.")
    parser.add_argument("--synthetic-only", action="store_true", help="Run only synthetic/control rows for fast development tests.")
    parser.add_argument("--freq-min-hz", type=float, default=5.0e9)
    parser.add_argument("--freq-max-hz", type=float, default=7.0e9)
    parser.add_argument("--center-freq-hz", type=float, default=None, help="Use one center frequency instead of a band.")
    parser.add_argument("--n-freqs", type=int, default=6)
    parser.add_argument("--cpml-layers", type=int, default=DEFAULT_CPML_LAYERS)
    parser.add_argument("--num-periods", type=float, default=DEFAULT_NUM_PERIODS)
    parser.add_argument("--dx", type=float, default=None, help="Optional grid spacing override for faster diagnostics.")
    parser.add_argument("--monitor-x-m", type=float, nargs="+", default=[0.030, 0.045, 0.060])
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Optional JSONL output path; default writes .omx/logs/wr90-port-oracle-matrix-<timestamp>-<mode>.jsonl; use '-' to print JSONL to stdout.",
    )
    parser.add_argument("--cw-warmup-cycles", type=int, default=20)
    parser.add_argument("--cw-lockin-cycles", type=int, default=20)
    args = parser.parse_args(argv)
    if not args.matrix:
        parser.error("--matrix is required")
    if len(args.monitor_x_m) < 3 and not args.synthetic_only:
        parser.error("at least three --monitor-x-m planes are required")
    if args.cpml_layers <= 0:
        parser.error("--cpml-layers must be positive")
    if args.num_periods <= 0:
        parser.error("--num-periods must be positive")
    if args.n_freqs <= 0:
        parser.error("--n-freqs must be positive")
    if not args.quick and not args.full:
        args.quick = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run_matrix(args)
    print(format_human_table(rows))
    if args.jsonl == "-":
        print("jsonl:")
        emit_jsonl_rows(rows, None)
        artifact_path = None
    else:
        mode = "full" if args.full else "quick"
        artifact_path = args.jsonl
        if artifact_path is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            artifact_path = f".omx/logs/wr90-port-oracle-matrix-{stamp}-{mode}.jsonl"
        path = emit_jsonl_rows(rows, artifact_path)
        print(f"jsonl_artifact: {path}")
    summary = {
        "n_rows": len(rows),
        "n_ok": sum(row.status == "ok" for row in rows),
        "n_control": sum(row.status == "control" for row in rows),
        "n_skipped": sum(row.status == "skipped" for row in rows),
        "n_error": sum(row.status == "error" for row in rows),
        "strict_closure_claimed": False,
        "issues_13_17_resolved": False,
    }
    print("summary:")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return 0 if summary["n_error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
