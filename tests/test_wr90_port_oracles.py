from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import scripts._wr90_port_oracle_matrix as oracle


def test_ref_free_gamma_solver_recovers_synthetic_three_plane_line():
    x = np.array([0.02, 0.04, 0.06])
    beta = 121.5
    a_plus = 1.2 - 0.3j
    gamma_true = 0.997 * np.exp(1j * 0.42)
    a_minus = gamma_true * a_plus
    u = a_plus * np.exp(-1j * beta * x) + a_minus * np.exp(+1j * beta * x)

    fit = oracle.solve_ref_free_gamma(x, u, beta)

    assert fit.rank[0] == 2
    assert fit.residual_norm[0] < 1e-12
    assert abs(fit.gamma[0] - gamma_true) < 1e-12


def test_ref_free_gamma_solver_handles_discrete_beta_band_and_noisy_overdetermined_fit():
    rng = np.random.default_rng(1234)
    x = np.array([0.018, 0.031, 0.047, 0.064, 0.079])
    beta = np.array([97.0, 123.0, 148.0])
    a_plus = np.array([1.0 + 0.1j, 0.9 - 0.2j, 1.1 + 0.05j])
    gamma_true = np.array([0.94 * np.exp(0.2j), 0.97 * np.exp(-0.3j), 1.01 * np.exp(0.1j)])
    samples = np.column_stack([
        a_plus[i] * np.exp(-1j * beta[i] * x)
        + gamma_true[i] * a_plus[i] * np.exp(+1j * beta[i] * x)
        for i in range(beta.size)
    ])
    samples += 1e-5 * (rng.normal(size=samples.shape) + 1j * rng.normal(size=samples.shape))

    fit = oracle.solve_ref_free_gamma(x, samples, beta)

    assert fit.gamma.shape == (3,)
    assert np.max(np.abs(fit.gamma - gamma_true)) < 1e-4
    assert np.max(fit.residual_norm) < 1e-4
    assert np.all(np.isfinite(fit.condition))


def test_integer_cycle_lockin_recovers_known_amplitude_and_phase():
    freq = 2.0e9
    samples_per_period = 64
    dt = 1.0 / (freq * samples_per_period)
    amp = 0.73
    phase = -0.61
    n = np.arange(samples_per_period * 9)
    signal = amp * np.cos(2.0 * np.pi * freq * n * dt + phase)

    phasor = oracle.integer_cycle_lockin(signal, freq, dt, start_index=samples_per_period, n_cycles=6)

    assert abs(abs(phasor) - amp) < 1e-12
    assert abs(np.angle(phasor * np.exp(-1j * phase))) < 1e-12


def test_emit_jsonl_rows_is_machine_readable(tmp_path):
    rows = [oracle.synthetic_least_squares_control(), oracle.cw_lockin_control_row()]
    path = tmp_path / "rows.jsonl"

    written = oracle.emit_jsonl_rows(rows, path)

    assert written == str(path)
    payload = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["status"] for row in payload] == ["control", "control"]
    assert payload[0]["case"] == "synthetic_ref_free_least_squares_control"
    assert payload[1]["case"] == "cw_lockin_synthetic_control"


def test_quick_matrix_contract_with_physical_rows_monkeypatched(monkeypatch):
    baseline = oracle.OracleRow(
        "baseline_current_2run_internal_mask_current_cpml",
        "current_2run",
        "ok",
        {"mean_abs_s11": 0.955, "monitor_backend": "production_two_run_waveguide_s_matrix"},
        "B_or_C_if_ref_free_good",
    )
    ref_free = oracle.OracleRow(
        "ref_free_3plane_internal_mask_current_cpml",
        "ref_free_multiplane",
        "ok",
        {
            "mean_abs_s11": 0.999,
            "monitor_backend": "passive_waveguide_ref_voltage",
            "monitor_distances_m": [0.02, 0.03, 0.04],
            "source_short_distance_m": 0.075,
            "short_type": "internal_mask",
            "cpml_layers": 10,
            "beta_type": "yee_discrete",
            "dft_window": "rect_full_record",
            "fit_residual": 1e-3,
            "fit_cond": 3.0,
        },
        "A_D_or_C3_if_ref_free_deficit_persists",
    )
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: baseline)
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ref_free)
    args = Namespace(
        synthetic_only=False,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=False,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
    )

    rows = oracle.run_matrix(args)

    cases = {row.case for row in rows}
    assert "baseline_current_2run_internal_mask_current_cpml" in cases
    assert "ref_free_3plane_internal_mask_current_cpml" in cases
    assert "synthetic_ref_free_least_squares_control" in cases
    ref_payload = next(row.to_jsonable() for row in rows if row.case == "ref_free_3plane_internal_mask_current_cpml")
    for key in (
        "monitor_backend",
        "monitor_distances_m",
        "source_short_distance_m",
        "short_type",
        "cpml_layers",
        "beta_type",
        "dft_window",
        "fit_residual",
        "fit_cond",
    ):
        assert key in ref_payload


def test_full_matrix_emits_explicit_deferred_rows(monkeypatch):
    ok = oracle.OracleRow("ref_free_3plane_internal_mask_current_cpml", "ref_free_multiplane", "ok", {"mean_abs_s11": 0.99})
    monkeypatch.setattr(oracle, "run_current_2run_baseline", lambda *args, **kwargs: oracle.OracleRow("baseline_current_2run_internal_mask_current_cpml", "current_2run", "ok"))
    monkeypatch.setattr(oracle, "run_reference_free_case", lambda *args, **kwargs: ok)
    monkeypatch.setattr(oracle, "run_source_purity_empty_line_sweep", lambda *args, **kwargs: oracle.OracleRow("source_purity_empty_line_sweep", "source_purity_line_sweep", "ok"))
    args = Namespace(
        synthetic_only=False,
        cpml_layers=10,
        num_periods=40.0,
        dx=None,
        full=True,
        monitor_x_m=[0.03, 0.045, 0.06],
        center_freq_hz=None,
        freq_min_hz=5.0e9,
        freq_max_hz=7.0e9,
        n_freqs=3,
        cw_warmup_cycles=20,
        cw_lockin_cycles=20,
    )

    rows = oracle.run_matrix(args)

    payload = [row.to_jsonable() for row in rows]
    cases = {row["case"] for row in payload}
    assert "pml_sweep_current_2run_layers_10_20_40" in cases
    assert "face_short_ref_free_no_cpml_or_irrelevant" in cases
    assert any(row["status"] == "skipped" and row.get("skip_reason") for row in payload)
