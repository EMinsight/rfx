"""GPU forward-throughput benchmark (successor to the retired 2026-07 harness;
same result schema: mc_per_s_median / spread per (cells, variant)).

Matrix: cube sizes 100^3..512^3 (skip on OOM, say so) x variants
  bare  — uniform vacuum cube, one soft source, no monitors
  nu-z  — same cube with a 4:1 graded dz_profile (the campaign's workload
          class exercises the NU lane; a uniform-only number overstates us)
fp32, CPML boundaries. Method: JIT+warmup excluded, then 3 timed windows of
N steps each (window sized ~4 s), median and spread reported.
Writes gpu_throughput_bench_results.json next to itself and prints it.
"""
import json
import os
import time

import numpy as np

import jax


def build(n, variant):
    from rfx import Simulation, GaussianPulse
    L = n * 1e-3
    kw = {}
    if variant == "nu-z":
        # 4:1 graded z: n cells whose sizes ramp 0.4..1.6 mm, sum == L
        r = np.linspace(0.4, 1.6, n)
        kw["dz_profile"] = r * (L / r.sum())
    sim = Simulation(freq_max=15e9, domain=(L, L, L), dx=1e-3,
                     boundary="cpml", cpml_layers=8, **kw)
    sim.add_source((L / 2, L / 2, L / 2), "ez",
                   waveform=GaussianPulse(f0=8e9, bandwidth=0.7),
                   amplitude_kind="current")
    return sim

def measure(n, variant):
    """Marginal-cost differencing: every ``sim.run(N)`` call pays a large
    fixed cost (material assembly, rasterization, host<->device transfer,
    dispatch) that has nothing to do with stepping.  The first version of
    this harness timed whole ``run()`` calls and read 152 Mcells/s where
    the same card's published measured number is 7,309 — the ladder ROSE
    with grid size, the signature of fixed-overhead domination.  So:
    time run(N_small) and run(N_big) and take
    cells*(N_big-N_small)/(t_big-t_small) — the setup cost subtracts out
    exactly, leaving the compiled scan's per-step cost.
    """
    sim = build(n, variant)
    t0 = time.perf_counter()
    r = sim.run(n_steps=8)            # compile both step counts' graphs? no:
    t_compile = time.perf_counter() - t0
    cells = int(np.prod(np.asarray(r.state.ez).shape))
    N1, N2 = 64, 1088                  # delta = 1024 scanned steps
    # warm both graph shapes once (scan length can retrigger compilation)
    jax.block_until_ready(sim.run(n_steps=N1).state.ez)
    jax.block_until_ready(sim.run(n_steps=N2).state.ez)
    meds = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(sim.run(n_steps=N1).state.ez)
        t1 = time.perf_counter()
        jax.block_until_ready(sim.run(n_steps=N2).state.ez)
        t2 = time.perf_counter()
        dt_steps = (t2 - t1) - (t1 - t0)
        if dt_steps <= 0:
            continue
        meds.append(cells * (N2 - N1) / dt_steps / 1e6)
    if not meds:
        raise RuntimeError("differencing produced no positive window")
    return dict(cells=cells, variant=variant, steps=f"{N1}->{N2}",
                mc_per_s_median=float(np.median(meds)),
                mc_per_s_spread=float(np.max(meds) - np.min(meds)),
                method=f"marginal cost, N={N1}->{N2}, 3 repeats",
                windows=len(meds), compile_s=round(t_compile, 2),
                jax_version=jax.__version__,
                device=str(jax.devices()[0]),
                dtype="float32")

def main():
    out = []
    for n in (100, 200, 300, 400, 512):
        for variant in ("bare", "nu-z"):
            try:
                row = measure(n, variant)
                print(f"{n}^3 {variant:5s}: {row['mc_per_s_median']:8.0f} "
                      f"Mcells/s (spread {row['mc_per_s_spread']:.0f}, "
                      f"steps {row['steps']})", flush=True)
                out.append(row)
            except Exception as e:  # OOM at large n on small cards — say so
                msg = f"{type(e).__name__}: {str(e)[:140]}"
                print(f"{n}^3 {variant:5s}: SKIPPED ({msg})", flush=True)
                out.append(dict(cells=n ** 3, variant=variant, skipped=msg))
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gpu_throughput_bench_results.json")
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
