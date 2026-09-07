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
