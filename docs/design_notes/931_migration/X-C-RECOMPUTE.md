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


---

## RESULTS (2026-09-07)

| case | verdict | evidence |
|---|---|---|
| cv14 | CONTROL HOLDS | mode table byte-identical to the pre-change baseline run; Gate 0 `max abs(eff - target) = 0.000e+00 m`, `eff = (50.000000, 30.000000, 40.000000) mm` — now MEASURED from `realized_wall_planes`, not computed from the grid shape. Gates 1-3 pass, worst mode error 0.0529 % (TE102) |
| cv16 | CONTROL HOLDS | all five gated rows byte-identical to the baseline apart from wall-clock seconds; `rfx_monostatic_dbsm` at ka=0.5 is -40.92, the committed fixture's value to 4 dp. The new `[CONDUCTOR MODEL]` line prints at every point |
| cv17 | CONTROL HOLDS | all four gated rows byte-identical to the baseline; `rfx -54.57 dBsm` at ka=0.5 matches the committed fixture. Dielectric sampling did not move |
| cv15 | MIGRATED, all six gates pass, and the shift is LARGE | f_primary 2.313947 → 2.436612 GHz (+5.30 %). Split by a decomposition run: conductor declarations +2.48 %, feed change +2.75 %. Both pre-declarations scored in `validation/crossval/_15_patch_results/RECOMPUTE.md` — the dip-depth prediction was right, the frequency-direction prediction was wrong, and the "most of it is the ownership term" prediction was wrong |

A fifth run, `369367259164`, is the cv15 feed decomposition
(`scripts/vessl_931_post_cv15_feed_decomposition.yaml`), writing
`_15_patch_results/rfx_decomposition_feed_pre931.json`.

Each case was accidentally submitted twice (a foreground submit loop was killed
by a tool timeout after it had already created runs). No VESSL run was deleted;
the `.latest` pointers name the second of each pair, and the two cv15 runs
agree to the last digit.
