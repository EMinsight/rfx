# X-D phase 2b — manifest digit refresh, claim_scope literals, and the pass-2 landing checklist

Group X-D (cv18, cv19), branch `feat/931-wr90-iris`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XD-wr90iris`, merged up to
`feat/931-lattice-ownership` at `770c4e6c`.

Two things live here. **§1-§2** are replacement text for
`validation/crossval/manifest.json` (cases 18 and 19), which this group does not own —
apply them the way the phase-2a replacement docs were applied. **§3** is the checklist
for landing pass 2, written so it can be executed without re-deriving anything.

Everything quoted below is measured. Pass-1 runs: cv18 **369367259159**, cv19
**369367259160**. Pass-2 runs submitted 2026-09-07 19:28/19:29 UTC: cv18
**369367259291**, cv19 **369367259293**. Source commits the pass-2 runs execute:
`901a3ad9` (cv18 gates) and `e861d280` (cv19 gates).

---

## 1. `manifest.json` case `18_wr90_iris_modematch`

### 1.1 Gate digits — the whole GATED sentence

| where | old | new | arithmetic |
|---|---|---|---|
| pooled fine gate | `0.04 abs` | `0.02 abs` | ceil(0.0106 × 1.5 = 0.0159) at quantum 100 |
| pooled fine envelope | `0.0232` | `0.0106` | measured, run 369367259159 |
| "every config within" | `0.023` | `0.011` | worst per-config gap 0.0106 |
| Richardson gate | `0.01 abs` | `0.01 abs` | ceil(0.0046 × 1.5 = 0.0069) → unchanged |
| Richardson envelope | `0.0051` | `0.0046` | measured |
| first-order ratios | `0.527-0.604` | `0.407-0.440` | recomputed per pair |

### 1.2 REPORTED digits

| where | old | new |
|---|---|---|
| coarse rung range | `0.018-0.043 abs` | `0.008-0.025 abs` |
| raw normalize=False gaps | `0.021-0.054` | `0.009-0.025` |
| pointwise \|raw−flux\| | `up to 0.033` | `up to 0.0068` |
| detrended ripple | `fine <= 0.0077, coarse <= 0.0158` | `fine <= 0.0076, coarse <= 0.0152` |

### 1.3 Modal-fence retraction digits

`1.0013-1.0207` → `1.0012-1.0200`. The four rows in order (d = 7.62 a/30, 7.62 a/60,
12.192 a/30, 18.288 a/30) are `1.0200 / 1.0099 / 1.0150 / 1.0012`, zero extractor
warnings. Modal |S11| gaps `0.0099-0.0460` → `0.0059-0.0269`; modal still runs a little
worse than flux at the same configurations, so the sentence's *argument* is unchanged.

### 1.4 APERTURE RESOLUTION — this paragraph INVERTS, it is not a digit swap

Per-config gates `0.019/0.034/0.015/0.022/0.035/0.015/0.034/0.034`
→ **`0.012/0.016/0.006/0.016/0.016/0.006/0.015/0.015`**, each
`ceil(that config's own gap × 1.5)` at quantum 1000; the pooled ceiling is `0.02`, not
`0.04`. All ten constants moved down.

The rest of the paragraph must be **rewritten, not re-digited**. Rebuilt
`aperture_resolution.json` on the corrected-thickness traces:

| summary key | pre-#931 | post-#931 |
|---|---|---|
| `over_aperture_detected` | 8 | 8 |
| `over_aperture_min_margin_x` | 1.77 | **1.623** |
| `under_aperture_detected` | 2 | **8** |
| `under_aperture_min_margin_x` | (key did not exist) | **2.608** |
| `under_aperture_max_margin_x` | < 1.5 | 3.587 |
| `under_aperture_scores_better_configs` | two d = 7.620 rows | **`[]`** |
| `nearest_offset_fine_cells_values` | `[0.5]` | **`[0.0]`** |
| `nearest_offset_is_positive_at_all_pairs` | true | **false** |
| `richardson_detected_either_sign` | 0 | 0 |

Suggested replacement sentence for the manifest, matching the script's claim_scope:

> A one-cell aperture error at each rung is detected in BOTH signs at every one of the
> eight configurations — over-aperture at worst 1.623× the gate, under-aperture at worst
> 2.608× — where before the contract the under-aperture sign was detected at only two of
> eight and scored BETTER than the undefected row at both d = 7.620 configurations. That
> asymmetry was not an aperture property: the committed fine trace's nearest oracle sat
> at d plus half a fine cell, which was the thickness deficit reading out on the aperture
> axis, and with realized thickness equal to drawn thickness the nearest oracle sits at
> the DECLARED d at all eight. The calibration this case supplies downstream is now
> aperture-resolved to one fine cell in BOTH signs, not only to +1.

The live-defect citation (`one_cell_defect_live.json`) keeps its shape but every digit in
it moves — it is re-run after cv18 pass 2 (see §3.4). Do not copy its current numbers.

### 1.5 The ONE-CELL VOLUME WITNESS clause — the criterion changed

The manifest currently says the witness is "gated on the t = 1 residual lying inside the
range the t = 2..8 rungs span". That criterion is **retired for vacuity**, and the
manifest must say so rather than quoting a gate that no longer exists.

Measured (run 369367259159, a/30, d = 12.192 mm), residual vs t in cells:

    t = 1  2      3      4      5      6      8
       0.0312 0.0246 0.0207 0.0179 0.0157 0.0139 0.0109

The residual is monotone decreasing, so t = 1 is the extremum for *every* possible
outcome — a perfect 0.0000 fails the old criterion too. It carries no information in
either direction. Replacement clause:

> The witness is gated on THICKNESS IDENTIFICATION: for each swept rung the lattice-blind
> oracle is evaluated at t−1, t and t+1 cells and the residual argmin must land on t. All
> seven rungs identify their own thickness (margins 1.21–1.34× over the runner-up). At
> t = 1 the t−1 alternative IS the pre-#931 realization — one wall, a zero-thickness
> screen — and it is 4.33× worse (0.1347 against 0.0311, with t+1 at 0.0400), so the
> contract's rule at one cell is decided by measurement rather than convention. The
> first-stated criterion (t = 1 inside the t = 2..8 range) is RETIRED as vacuous and its
> verdict is kept in the record under `one_cell_volume_witness.monotone_range_criterion`.

New fixture keys under `one_cell_volume_witness`: `rows[*].identification`
(`gap_at_t_minus_1`, `gap_at_t`, `gap_at_t_plus_1`, `argmin_t_cells`,
`identified_own_thickness`, `margin_vs_runner_up_x`), `identified_every_thickness`,
`one_cell_two_wall_vs_one_wall_x`, `monotone_range_criterion`. `passed` now tracks the
identification verdict.

### 1.6 INGEST STATE

Delete the pass-1 provisional paragraph and replace with: regenerated on VESSL
**369367259291** (pass 2, source `901a3ad9`); pass 1 was 369367259159 and exited 1 by
design.

---

## 2. `manifest.json` case `19_wr90_iris_filter_aghanim`

### 2.1 Gate digits

| where | old | new |
|---|---|---|
| `gates.f0_measured_envelope_mhz` | `12.1230` | **`12.1219`** |
| `gates.f0_gate_mhz` | `19.0` | **`19.0` (unchanged — ceil(12.1219 × 1.5) = 19.0)** |
| `gated_rfx.d_f0_mhz` | `12.08` | **`12.12`** |
| population spread | `0.06 MHz` | **`0.02 MHz`** |
| "every member's \|d_f0\| is about" | `12.08` | **`12.12`** |

The f0 gate not moving is the pre-declared falsifier passing, and the manifest should say
so in those words.

### 2.2 GATING POSTURE — decided: edges and bandwidth are now GATED

The merge-time text ("band edges and bandwidth stay REPORTED … at ingest, either gate
them at round-UP(measured envelope × 1.5) or restate here why not") is discharged. The
decision is to **gate them**, and the arithmetic is:

| gate | envelope over the 9-config population | derivation | value |
|---|---|---|---|
| `GATE_EDGE_MHZ` | max(\|d_lo\|, \|d_hi\|) = **17.0553 MHz** | ceil(17.0553 × 1.5 = 25.583) | **26.0** |
| `GATE_BW_MHZ` | \|d_bw\| = **9.9024 MHz** | ceil(9.9024 × 1.5 = 14.854) | **15.0** |

Both halves of the old objection are answered by measurement, not by argument: the
half-cell comparator-input uncertainty is what the contract removes, and the regenerated
population supplies the envelope the gate was waiting on. The population is the SAME nine
configurations and the SAME settling criterion the f0 gate uses; `d_bw ≡ d_hi − d_lo`, so
the two gates are one fact read twice and neither corroborates the other. Read them as
regression locks with 50 % headroom, exactly like f0.

Replace `REPORTED, NEVER GATED: individual band edges (+17.08 / +7.09 MHz) and bandwidth
(-9.99 MHz), which are ONE fact and not two …` with the newly-gated form; the measured
gated-row values are **+17.05 / +7.20 MHz** and **−9.85 MHz**.

`gates` key renames (meaning changed, so the names changed):
`edge_reported_residual_mhz` → `edge_measured_envelope_mhz`,
`bw_reported_residual_mhz` → `bw_measured_envelope_mhz`, plus new `edge_gate_mhz`,
`bw_gate_mhz`, `edge_bw_envelope_population` (nine rows). The ring-down and b-invariance
rows gain `lo` / `hi`, so the population reads recorded band edges rather than ones
reconstructed from f0 and bw.

### 2.3 Other measured digits

| where | old | new |
|---|---|---|
| worst in-band RL, rfx | `9.80 dB` | `9.84 dB` |
| gated span | `340 MHz` | `341 MHz` |
| a/60 f0 residual | `+19.85 MHz` | `+19.87 MHz` |
| b-invariance b=4 vs b=8 | `152 Hz` | `233 Hz` |
| FDFD worst unitarity | `4.6e-07` | `5.0e-07` |
| rfx − FDFD | `+13.17 / -10.97 MHz` | `+13.21 / -10.83 MHz` |
| "as it differs from the cascade" | `(+12.08 / -9.99)` | `(+12.12 / -9.85)` |
| cell-equivalent residual | `-0.1169 / -0.1241 cell` | `-0.117 / -0.124 cell` |

Unchanged and worth stating as unchanged: the FDFD leg reproduces — `richardson_34`
f0 **10.95742 GHz**, BW **351.42 MHz**, consistency 0.37/0.36 MHz, level ratios 1.55 and
1.33, `empty_s11` 5.0e-14 — which is the geometry-preserving claim measured rather than
asserted. Also unchanged: coarse a/60 `16 of 24` bins, trough `2.73 dB`, colpow `1.0065`
gated and `1.207` excluded, oracle-to-oracle RL `13.82 → 10.65 dB`, the CST/HFSS anchor
scalars (oracle-side, no lattice dependence).

### 2.4 Sentences that are now HISTORY

`iris_thickness_zero_count_sweep` is re-centred (7.50–8.50 cells, eleven points) and the
count stays invariant at 3 while bandwidth moves **+41.6 MHz** across the band — the old
text says 20 MHz over the one-sided 8.00–8.50 window. The `-0.68` fit clause and the
`+107.5 MHz` bias clause stay as written; they are already labelled history.

### 2.5 INGEST STATE

Replace with: regenerated on VESSL **369367259293** (pass 2, source `e861d280`); pass 1
was 369367259160, exited 0, and its envelopes are what the constants were derived from.

---

## 3. Landing checklist for pass 2

Nothing below may be done by editing a number in a committed artifact. Every file listed
is either produced by the run or derived by a committed builder from a produced file.

### 3.1 Poll, do not `vessl run list`

    /root/workspace/claude-workspace/rfx/runs/issue931-post-cv18-<ts>/{cv18.rc,cv18.log,produced/}
    /root/workspace/claude-workspace/rfx/runs/issue931-post-cv19-<ts>/{cv19.rc,cv19.log,produced/}

`.latest` in the runs directory names the current output dir. **Expected rc is 0 for
both.** If a log carries an `ENVELOPE/GATE MISMATCH` line the constants in `901a3ad9` /
`e861d280` are wrong for the merged core and a pass 3 is needed with the printed values —
do not edit the fixture.

### 3.2 Commit the produced records

From the worktree (the runs write in place, so the files are already there; the run's
`produced/` copy is the cross-check):

    validation/crossval/_18_wr90_iris_results/rfx.json
    tests/fixtures/wr90_iris_modematch/fixture.json
    validation/crossval/_19_iris_filter_results/rfx.json
    tests/fixtures/wr90_iris_filter/fixture.json

### 3.3 Rebuild the aperture-resolution artifact (no FDTD, seconds)

    JAX_PLATFORMS=cpu python scripts/diagnostics/build_cv18_aperture_resolution.py

then `--check` must print OK. Commit
`validation/crossval/_18_wr90_iris_results/aperture_resolution.json`. Expected summary is
§1.4's post-#931 column; it was already rebuilt against the pass-1 record and reproduced
those values, so a difference here means the pass-2 traces moved and §1.4 must be redone
from the new artifact.

### 3.4 Re-run the migrated defect probe (VESSL, ~15 min, third run)

Only AFTER cv18 pass 2 — it reads `GATE_FINE_ABS_PER_CONFIG` from the script source and
the pooled/Richardson gates from the regenerated record:

    JAX_PLATFORMS=cpu python scripts/diagnostics/probe_cv18_one_cell_aperture_defect.py \
        --output validation/crossval/_18_wr90_iris_results/one_cell_defect_live.json

Six pinned digits in
`test_live_one_cell_defect_is_caught_by_the_per_config_gate_and_not_the_old_ones`
(0.02842, 0.00588, 0.015, 0.04, 0.01, 0.0265) ALL move: the per-config gate at
d = 7.620|0.20|0.50 is now **0.006**, the pooled gate **0.02**, the Richardson gate
0.01 unchanged, and the modelled row it sits beside is
`aperture_resolution.json::pairs[2].one_cell_defect.over` (`fine_gap_abs` 0.0134,
`fine_margin_x` 2.233). Re-pin from the new artifact in the same commit.

### 3.5 Test re-pins — the values are already measured, listed here so the landing
commit does not have to re-derive them

`tests/crossval/test_wr90_iris_modematch_gates.py`

| assertion | pre-#931 | post-#931 |
|---|---|---|
| `g["fine_gate_abs"] == 0.04` | 0.04 | **0.02** |
| `g["richardson_gate_abs"] == 0.01` | 0.01 | 0.01 |
| ripple pins `"0.0077"` / `"0.0158"` | | **`0.0076` / `0.0152`** |
| `"fine <= 0.0077, coarse <= 0.0158" in scope` | | **`fine <= 0.0076, coarse <= 0.0152`** |
| raw-vs-flux `approx(0.033)` and `"up to 0.033"` | | **`0.0068`** |
| coarse range `"0.018"` / `"0.043"`, `"0.018-0.043 abs"` | | **`0.008` / `0.025`, `0.008-0.025 abs`** |
| modal `max(colpow) == approx(1.0207)` | | **`1.0200`** |
| `script_gates[7.62\|0.2\|0.5] == 0.015` (×2 sites) | 0.015 | **0.006** |
| `sum(v <= pooled/2) == 3` | 3 | **2** (gates ≤ 0.01 are the two d = 7.620 rows) |
| audit defect `fr["max_gap_abs"] approx 0.0097` | | **0.0034** |
| audit defect `cr["richardson_dev_abs"] approx 0.0010` | | **0.0012** |
| audit defect `gap approx 0.0265` / `rich approx 0.0030` | | **0.0134 / 0.0027** |
| audit defect `gap/cfg_gate >= 1.7` | | **2.23** measured (bound may stay 1.7) |
| `detected[+1]` count 8, `min approx 1.77` | | **8, 1.623** |
| `detected[-1]` count 2, `max < 1.5` | | **8, min 2.608** |
| `under_aperture_detected == 2` | 2 | **8** |
| `under_aperture_max_margin_x < 1.5` | | **retire; use `under_aperture_min_margin_x >= 1.5`** |
| `nearest_offset_fine_cells_values == [0.5]` | | **`[0.0]`** |
| `nearest_offset_is_positive_at_all_pairs is True` | | **False** |
| `closer_than_nominal_wide == [two 7.620 rows]` | | **`[]`** |
| `dist["-1.0"] > dist["+0.0"]` | holds | still holds (0.0098–0.0342 vs 0.0034–0.0106) |
| `test_one_cell_volume_witness…` range assertion | | **replace with the identification assertions**, see §1.5 |

Keep both branches (pre-/post-#931) selected by `_fixture_is_post_931`, the way the
merge already did for the key-tolerant accessors; a single-valued edit would red the
tests between this commit and the fixture landing.

`tests/crossval/test_wr90_iris_filter_gates.py`

- `_PIN_F0_ENV_MHZ` `12.1230` → **`12.1219`**; `_PIN_F0_GATE_MHZ` stays `19.0`;
  `_PIN_GATED_MAX_COLPOW` stays `1.0065`; `_PIN_COARSE_MAX_COLPOW` stays `1.0315`.
- `_PIN_TRACE_SHA256` — recompute from the pass-2 fixture, in that commit. It is the one
  value that cannot be prepared in advance.
- `_PINS_REPINNED_FOR_931 = False` → **`True`**, in that same commit and no earlier;
  the flag exists so the pins skip rather than red while the fixture is pre-#931.
- Add pins for the new gates: `edge_gate_mhz == 26.0`, `bw_gate_mhz == 15.0`, each with
  the derived relation `gate_from_envelope(env, quantum=1)` recomputed from
  `edge_bw_envelope_population` — hard pin AND derived relation, both, the way the f0
  gate is pinned; and assert the population is nine rows and the same configs as
  `f0_envelope_population`.
- `test_non_gated_quantities_are_declared_non_gated` still passes: the posture keeps the
  literal phrase "band edges and bandwidth", now on the GATED side.

### 3.6 Not this group

- `validation/crossval/manifest.json` cases 18/19 — §1 and §2 above.
- `docs/public/guide/benchmarks.mdx` line 67 — see `crossval-D-benchmarks.mdx.md`.
- `tests/data/example_fidelity_snapshot.json` — the merge agent regenerates it last.
