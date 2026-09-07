# X-C recompute index — issue #931, cases cv14 / cv15 / cv16 / cv17

Branch `feat/931-crossval-C`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XC-crossvalC`.
One VESSL run per case, preset `gpu-rtx4090`, cluster `remilab-c0`,
`JAX_PLATFORMS=cpu`. Per-case detail lives in each results directory:

* cv15 — `validation/crossval/_15_patch_results/RECOMPUTE.md`
* cv16 — `validation/crossval/_16_ka_sweep_results/RECOMPUTE.md`
* cv17 — `validation/crossval/_17_dielectric_results/RECOMPUTE.md`
* cv14 — this file (cv14 commits no artifact; its gate table and exit code
  are the record, so it has no results directory)

| case | run id | yaml | command | expected runtime | verdict wanted |
|---|---|---|---|---|---|
| cv14 | `369367259152` | `scripts/vessl_931_post_cv14.yaml` | `python -u validation/crossval/14_rect_cavity_pozar.py` | seconds | CONTROL — all four gates reproduce, and the new `[WALL REGISTRATION]` line reports realized planes `x=[0,50] y=[0,30] z=[0,40]` |
| cv15 | `369367259156` | `scripts/vessl_931_post_cv15.yaml` | `… 15_patch_antenna_rt5880.py rfx --num-periods 45.0 --gain`, then `… compare` | 5-8 min | MIGRATION — the leg is regenerated; gates must pass and the stack check must report `sheet / sheet` |
| cv16 | `369367259153` | `scripts/vessl_931_post_cv16.yaml` | `python -u validation/crossval/16_pec_sphere_mie_ka_sweep.py` | 5-10 min | CONTROL — fenced sigma fill, every gated `delta_db` / `a_eff_over_a` reproduces. **No** `--write-fixture` |
| cv17 | `369367259154` | `scripts/vessl_931_post_cv17.yaml` | `python -u validation/crossval/17_dielectric_sphere_mie.py` | 5-10 min | CONTROL — dielectric-only, bit-identical. **No** `--write-fixture` |

Submitted 2026-09-07 10:07-10:09 UTC, organization `remilab`, project
`byungkwan`; run pages at `https://app.vessl.ai/remilab/runs/byungkwan/<run id>`.
Submitted from a non-worktree directory: the `vessl` CLI walks the cwd's `.git`
as a directory and a git worktree's `.git` is a file, so `vessl run create`
raises `NotADirectoryError` if run from inside one.

Outputs: `/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<UTC>/`,
with a `…-<case>.latest` pointer beside it. Each job harvests every file newer
than a start marker into `$OUT/produced/`.

cv14, cv16 and cv17 run in a container-local staged copy (`cp -a`) and write
nothing anywhere. cv15 runs IN the worktree so the regenerated leg lands on the
branch for the ingest phase; that is the only job that touches the checkout.

## cv14 — why a control run at all

cv14's only PEC is the domain boundary, which the contract fences (§1.8) and
leaves at E_tan = 0 on node planes 0 and n-1. Nothing measured that before:
Gate 0, named "wall registration", computed `(n-1)*dx` from the grid SHAPE and
would have kept passing under any change to how a PEC wall is realized. Gate 0
and the Yee oracle now read the MEASURED planes (`apply_pec` / `apply_pec_faces`
applied to an all-ones state, planes read back through `realized_wall_planes`),
with `WALL_REG_TOL_M` unchanged at 1e-9 m — a strict tightening. The run is the
physics leg; the falsifier (flip one face to PMC and the assertion raises) is
build-time and already exercised.
