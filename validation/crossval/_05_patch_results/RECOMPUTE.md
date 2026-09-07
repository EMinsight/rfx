# cv05 + dielectric controls — recompute record (#931, group X-A)

Branch `feat/931-crossval-a`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-XA-crossval-a`.

Everything here re-solves; nothing is translated from an old number. Each job
reads THIS worktree in place over the NFS mount (not a staged copy), so the
ring-down fixture the cv05 producer writes lands on the branch and the cv04
control can be a `git diff` against the committed artifacts.

Submitted 2026-09-07, cluster `remilab-c0`, preset `gpu-rtx4090`,
`JAX_PLATFORMS=cpu` inside the job (these are CPU-produced fixtures; the preset
is the machine allocation, not a device choice).

| case | VESSL run | yaml | what it does | expected wall clock |
|---|---|---|---|---|
| cv05 | **369367259142** | `scripts/vessl_931/cv05.yaml` | realization assert (no solve) → all three parts with the image's openEMS → geometry-only census → ring-down fixture rebuild | 1.5–3 h (baseline PART 1 alone was 158.1 s; the fixture rebuild is 5 lengths × 2 solves) |
| cv04 | **369367259144** | `scripts/vessl_931/cv04.yaml` | re-run + `git diff --exit-code` on the two committed JSONs | 10–30 min (8000 steps, hand-written loop; it exceeded 2 min on the shared pod, which is why it is here) |
| cv01 | **369367259145** | `scripts/vessl_931/cv01.yaml` | two rfx runs (straight + bend), Meep absent → exit 2 | 20–40 min |
| cv03 | **369367259146** | `scripts/vessl_931/cv03.yaml` | rfx leg + slab TE0 oracle gate, Meep absent → exit 2 | 20–40 min |

Submit command (run it from OUTSIDE a git worktree — the vessl CLI opens
`.git/HEAD` as a path and a linked worktree's `.git` is a FILE, so
`vessl run create` raises `NotADirectoryError` when cwd is inside one):

    cp scripts/vessl_931/cv05.yaml /tmp/x/ && cd /tmp/x && vessl run create -f cv05.yaml

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<UTC>/`,
with the newest path echoed to `issue931-post-<case>.latest`.

## Pre-declared, before any of the four finished

Written down now so the numbers can be adjudicated instead of merely reported.

**cv05 will move away from its committed 6.48 % rfx-vs-openEMS agreement, and
that is not a regression.** Two records say so independently:

* `rfx-known-issues.md`, 2026-08-28 A/B verdict: "two_plane is the ONLY
  realization that gives the physically correct cavity ... and it is the
  realization that agrees WORST with the external reference ... Exactness of
  the cavity and agreement with the reference point in opposite directions."
* research note 20260711: a six-cell substrate next to a fine↔coarse grading
  transition split the mode and took the openEMS agreement from 2.65 % to
  6.45 %. The transition still sits immediately outside the fine block here
  (the z-profile builder grades the air, not the substrate), so that risk is
  live and unmitigated. It is reported, not designed around.

Direction: the realized cavity goes from 1.8161 mm (of which the top 455 µm was
air) to the declared 1.5 mm of FR4. A thinner, fully-loaded cavity raises the
effective permittivity seen by the patch and lowers the resonance; the
committed `rfx_harminv_hz` 2.331854551 GHz sat 3.78 % BELOW the analytic
2.423509825 GHz, so the expectation is that `rfx_vs_analytic_pct` grows in
magnitude on the same (negative) side. **A sign flip would mean something other
than the cavity moved and must be investigated before the number is quoted.**

The window it is judged against is measured, not asserted: the
`RFX_CV05_SHEET_PLANE_DELTA` arm (ground down one node plane, patch up one)
realizes a 2.1500 mm cavity against the declared 1.5000 mm, i.e. one node plane
per wall is worth +43 % of cavity height. The pre-contract error was +21.1 %,
about half of that, so a resonance shift of the order of the half-arm is
expected and anything far outside it is the thing to explain.

**cv01 / cv03 / cv04 must not move at all.** They are dielectric-only. This is
already established at build time — `scripts/diagnostics/cv0104_dielectric_control_witness.py`
digests the assembled `eps_r` / `sigma` arrays as raw bytes and compares this
branch against the pre-#931 checkout `rfx-baseline-d990e18c`:

    01_waveguide_bend    [181,181,1]  eps a693e67131469a9d  sigma 81205f6f74a3487c  IDENTICAL
    02_ring_resonator    [162,162,1]  eps 40dbec5bbf922728  sigma e006234d0697ae9c  IDENTICAL
    03_straight_wg_flux  [201,131,1]  eps f12b4af1ea61604e  sigma 309d6171b688ed09  IDENTICAL

all three with 0 PEC cells, 0 sheets, 0 wires on both sides — conductor-free by
measurement, not by reading the source. The four jobs above are the end-to-end
confirmation of the same claim. **cv04's job FAILS if its two committed JSONs
change by a byte.**

cv02 has no post job: no pre-change baseline job exists for it and no artifact
of it is committed, so a post-only run would have nothing to be compared
against. Its control is the digest row above, which compares bit patterns and
is stronger evidence than a solve.

## Fixture keys that will change

`tests/fixtures/patch_mode_identification/cv05_ringdown_spectra.json`, rebuilt
by `scripts/diagnostics/build_cv05_ringdown_spectra.py` inside the cv05 job:

* `runs.*.modes[*].freq` / `.Q` / `.amplitude` — every one; the cavity changed.
* `runs.*.realized_stack` — NEW. The planes, the cavity and the footprints of
  the build that produced each row, from `realized_pec_edge_masks`. A fixture
  whose provenance is a resonance and nothing else cannot say which cavity
  produced it; that is how the +21.1 % survived several re-pins.
* `_realized_x_cell_census` — re-measured, not carried forward, and its UNIT
  changes with the measurement. The old block counted masked NODES; the
  realized conductor is the tangential E edges its footprint stands on, and
  28 mm of metal is what sets the resonance, not 29 nodes. Measured locally,
  geometry only: 29.5 mm declared → **28** edges (was 29 nodes); 22.0 mm → 22
  either way, because both its faces are on the lattice. Expect every
  off-lattice row to drop by one and every on-lattice row to hold.
* `_provenance` — new commit, new timestamp.
* `_declared_TM100_hz`, `_what`, `_settling` — unchanged; `_declared_TM100_hz`
  is a closed form in εr / h / L / W and is mesh-independent.

`validation/crossval/_05_patch_results/cv05_run_openems_369367257743.json`:
every `rfx_*` key is a measurement of the old realization and is regenerated,
not adjusted. `openems_*` and `declared_modes_hz` are unaffected by the rfx
contract, which makes the old file the natural before/after witness — keep it
beside the new one.

## What ingest still has to do

1. Place the new cv05 run record beside the old one (do not delete
   `cv05_run_openems_369367257743.json`; it is the "before" half of the
   evidence) and update `manifest.json` from
   `docs/design_notes/931_migration/XA-manifest.md`.
2. Commit the rebuilt `cv05_ringdown_spectra.json` — this phase deliberately
   does not.
3. Re-derive the farfield envelope band from the new run per the rule stated in
   the manifest note. It is fitted to a realization, so it cannot be translated.

---

# MEASURED (2026-09-07, all four jobs)

Written against the pre-declaration above, verbatim, pass or fail.

## Controls — cv01, cv03, cv04: unchanged, as declared

* **cv04 (369367259144): CONTROL PASS.** `git diff --exit-code` on
  `validation/crossval/_04_fresnel_results/` is EMPTY. Both committed JSONs
  (`fringe_gate_geometry.json`, `lattice_witness.json`) reproduce byte-for-byte
  from a re-run of the migrated checkout. "rfx accuracy: PASS" in the log.
* **cv01 (369367259145) and cv03 (369367259146):** the post-change logs are
  IDENTICAL to the pre-change baseline logs (369367258*, run 05:16 / 05:22) line
  for line, apart from one wall-clock line in cv03 ("Done in 2.0s" → "2.6s").
  Every printed digit holds: cv01 straight self-T 0.9892, smoothed T 0.6374;
  cv03 two-wave residual 0.0077, n_eff band mean rfx 2.84634 / analytic 2.84203,
  max |n_eff deviation| 0.262 % at f = 0.1592 (rfx 2.89607 vs analytic 2.88849),
  band-mean T 0.9657. cv03's 2 % n_eff gate against a closed-form slab TE0
  oracle is the sharpest detector of a dielectric Box occupancy change in this
  group, and it did not move by a digit.
* cv02 has no solve, by the reasoning above; its build-time material digest is
  bit-identical to the pre-#931 checkout.

The claim "dielectric sampling is untouched" holds on every control, at both the
array level and end to end.

## cv05 (369367259142) — migrated, and the pre-declaration was WRONG on sign

| quantity | committed (VESSL 369367257743) | post-contract | |
|---|---|---|---|
| realized cavity | 1.8161 mm node-to-node, top 455 µm air | **1.5000 mm, six FR4 cells** | as declared |
| realized wall planes | 12.0000 / 13.8161 mm (argmin) | **12.0000 / 13.5000 mm (k = 22 / 28)** | declared |
| ground footprint | 58 × 54 mm (drawn 60 × 55) | **60.0 × 55.0 mm** | drawn |
| patch footprint | 28 × 37 mm (drawn 29.5 × 38) | 28.0 × 37.0 mm | unchanged; half-cell debt |
| `rfx_harminv_hz` | 2 331 854 551 | **2 446 496 652** | +4.92 % |
| `rfx_vs_analytic_pct` | −3.7819 | **+0.9485** | **SIGN FLIPPED** |
| `rfx_vs_openems_harminv_pct` | 6.4774 | **11.7122** | worse, as predicted |
| `rfx_internal_pct` | 0.5084 | 1.3694 | still inside the 5 % self-consistency gate |
| `rfx_s11_dip_hz` / depth | 2.32 GHz / −1.61 dB | **2.48 GHz / −9.62 dB** | the feed now couples |
| settling witness | −21.6 dB (UNDERSETTLED) | −20.4 dB (UNDERSETTLED) | unchanged caveat |
| `status` | failed | passed | |

**The prediction that failed.** RECOMPUTE.md said, before the run: "the
expectation is that `rfx_vs_analytic_pct` grows in magnitude on the same
(negative) side. A sign flip would mean something other than the cavity moved
and must be investigated before the number is quoted." It flipped, and the
magnitude shrank. The prediction was wrong, and it was wrong for a reason that
was already written down in the memory this note cites: it modelled only the
cavity HEIGHT and ignored the two other mechanisms the same change removes,
both of which push the resonance UP —

* the #702 own-cell resample, which rewrote the ground sheet's own cell layer
  from vacuum to dielectric INSIDE the cavity (`rfx-known-issues.md`: "18590
  cells, ALL on z-plane k=29, all 1.000 → 3.380 ... the whole ground-plane cell
  layer, which sits INSIDE the patch cavity", worth ~13.5 pp on the canonical
  patch). Deleting it lowers the cavity's effective permittivity and raises f;
* the 455 µm vacuum layer that sat directly under the old patch plane, which is
  now FR4.

The same memory entry says outright that "the historical 9.32 was not correct —
it was two errors of opposite sign cancelling". A one-mechanism prediction
should not have been written after quoting that sentence. Recorded, not
repaired: no second cv05 solve was run to chase the prediction.

**What was predicted correctly.** The openEMS agreement got worse, 6.48 % →
11.71 %, exactly the direction the 2026-08-28 A/B verdict gives ("exactness of
the cavity and agreement with the external reference point in opposite
directions"). It stays inside the case's 20 % smoke bound. Note that the
openEMS number it is measured against, 2.19 GHz, is one the case itself refuses
to gate: `openems_mode_id_ok` is false on both runs because openEMS's
port-voltage ring-down carries poles that claim several declared members at
once.

**R5 — the metric is not quoted alone.** The full ring-down spectrum, not the
headline, and it reorganized in the direction a corrected cavity predicts:

| declared member | committed run | post-contract run |
|---|---|---|
| TM010 (1.914913 GHz) | **not found** | 1.900347 GHz, **−0.76 %** |
| TM100 (2.423510 GHz) | 2.331855 GHz, −3.78 % | 2.446497 GHz, **+0.95 %** |
| TM110 (3.088737 GHz) | 3.027300 GHz, −1.99 % | 3.186911 GHz, **+3.18 %** |
| unidentified poles | 3.605107 GHz | 3.758955 GHz |

Three measured poles now identify against the declared TM_mn0 spectrum instead
of two, and the member that was missing entirely is found within 0.8 %. That is
a structural agreement with the declared cavity, not a single number moving.
The second independent witness is the port: the S11 dip deepened from −1.61 dB
to −9.62 dB, which is the port-extent fix (it now spans realized wall to
realized wall rather than stopping at a drawn coordinate a node plane below the
patch) showing up as coupling. The two witnesses do not share a suspect
quantity.

**Caveat that did NOT improve:** the ring-down is still UNDERSETTLED (−20.4 dB
against a −40 dB bar), as it was before (−21.6 dB). Harminv frequency and Q
carry truncation error on both runs equally. Raising `num_periods` is a
separate change and was not made here; it must not be folded into this
migration.

## Census, re-measured (geometry only, no FDTD)

| declared L (mm) | committed (nodes) | post-contract (edges) |
|---|---|---|
| 29.5 | 29 | 28 |
| 23.0 | 23 | 22 |
| 22.5 | 23 | 22 |
| 22.0 | 22 | 22 |
| 21.65 | 21 | 20 |
| 21.5 | 21 | 20 |
| 21.0 | 21 | 20 |
| 20.5 | 21 | 20 |
| 38.0 | 38 | 38 |

Every row is exactly one lower except 22.0 and 38.0, whose faces are on the
lattice. This is the unit change (masked nodes → realized metal edges), not a
geometry change: N masked nodes have always carried N−1 tangential edges, and
the realized conductor is the edges. The rows that hold are the ones where the
old node count already equalled the edge count.

## Still open at the time of writing

The ring-down fixture rebuild (`cv05_ringdown_spectra.json`, 5 lengths × 2
solves) was still running inside run 369367259142 when this section was
written; it writes into this worktree. Ingest takes it from
`git status` / the run's `produced/` directory. Its `runs.*.modes` will follow
the spectrum above, and the manifest's three cited fixture values must be
re-read from it per `docs/design_notes/931_migration/XA-manifest.json.md` §3 —
including re-checking that the 22.0 mm build is still assigned TM110, which is
what makes criterion B a demonstration rather than a claim.
