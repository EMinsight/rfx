# E — examples: recompute record (#931 lattice ownership contract)

Branch `feat/931-examples`, worktree
`/root/workspace/byungkwan-workspace/research/rfx-931-E-examples`.
All runs submitted at commit `7e0de966ad736606d3c6dbea2f0bd6d9b670d9de`
(the second of this group's three commits; the third adds only these notes).

Machine rule: nothing longer than ~1 minute is solved on the shared pod. Every
re-solve is one VESSL run per case, preset `gpu-rtx4090` with
`JAX_PLATFORMS=cpu` inside the job (these fixtures are CPU-produced).
YAMLs live in the worktree at `.vessl931/` (untracked) and were submitted from
a non-git directory — `vessl run create` walks up for a git repo and dies on a
worktree's `.git` FILE with
`NotADirectoryError: .../.git/HEAD`, so submit from outside the checkout.

| case | run id | script | expected runtime |
|---|---|---|---|
| ex-patch-demo | **369367259175** | `examples/tutorials/patch_antenna_demo.py` | ~22 min CPU (pre-change measurement 1345 s) |
| ex-nu-patch-demo | **369367259177** | `examples/tutorials/nonuniform_patch_demo.py` | ~19 min CPU (pre-change measurement 1169 s) |
| ex-tutorials | **369367259181** | the 12 tutorial / quickstart scripts end to end plus `rfx run examples/config/microstrip_thru.yaml` | ~30-40 min total |

Submit command used, per case:

```
cd <a directory that is NOT inside a git checkout>
vessl run create -f <case>.yaml        # retried 3x on Bad Gateway
```

Outputs land in `/root/workspace/claude-workspace/rfx/runs/issue931-post-<case>-<ts>/`,
with the latest path echoed to `issue931-post-<case>.latest`. Each job harvests
every file the run wrote into its staged checkout under `produced/`.

## What each run is for

**ex-patch-demo / ex-nu-patch-demo** — the two patch tutorials record measured
numbers in their docstrings and printed output (runtime, settling dB, the
harminv mode list, the deviation against openEMS 2.4221 GHz). Those numbers
are still the PRE-migration ones in the committed files: the migration removed
a vacuum cell from inside the realized cavity (patch demo solved a 1.905 mm
board against a declared 1.524 mm; the NU demo 2.0 mm against 1.5 mm), so
every recorded frequency is expected to move DOWN. The ingest phase replaces
the recorded numbers from these logs with the pre-change values quoted beside
them.

Pre-change baselines for the same two cases were submitted by the baseline
agent from `scratchpad/vessl_baseline/{ex-patch-demo,ex-nu-patch-demo}.yaml`
(ids in `scratchpad/vessl_baseline/submitted_ids.txt`); compare against those,
not against the docstrings.

Declared falsifier for both, from design note §5 (cv15 arm), already asserted
at build time and measured on this pod without solving:

* `patch_antenna_demo`: sheets at z-node 29 / 33, realized wall planes
  `[29, 33]`, node-to-node cavity 1.524 mm = declared `SUB_THICK`.
* `nonuniform_patch_demo`: sheets at z-node 33 / 39, realized wall planes
  `[33, 39]`, node-to-node cavity 1.500 mm = declared `h_sub`.

If a re-solve comes back with the resonance UNMOVED, the diagnosis is wrong:
the cavity changed by 25 % (patch) and 33 % (NU) in vacuum-cell terms, so the
frequency must move.

**ex-tutorials** — the zero-advisory gates and the migrated example scripts,
run where the box is not shared. Three of them
(`rcs_scattering`, `run_control_and_fields`, `materials_and_dispersion`) fail
`tests/contracts/test_tutorial_examples.py` on this pod for a reason that is
NOT the migration: that test runs each tutorial through `subprocess` with a
60 s timeout, and on this loaded pod `ports_and_sparams_101` alone takes 101 s
BEFORE the migration and 110 s after (measured, both). Two of the three
failures are on files this group never touched. The timeout was left alone;
this run is the evidence that the scripts themselves are healthy.

Checked directly on the pod, without VESSL, before submitting:

| script | result |
|---|---|
| `ports_and_sparams_101.py` | rc 0; realized z wall planes `[8, 12]` = 1.25 / 2.25 mm, strip-to-ground 1.00 mm; generic-port S11 bit-identical to pre-change; `All checks passed` 4 -> 3 |
| `slab_rt_flux_monitor.py` | rc 0; material arrays bit-identical to the nudged version (290 cells) |
| `rcs_scattering.py` | rc 0; PEC volume 1067 cells / sigma fill 1045 / analytic 1079.8; backscatter unchanged (the sigma path is fenced, §1.8) |
| `resonance_harminv.py` | rc 0; TE101 7.505940 GHz, error 0.005706 % — boundary-face PEC control, unchanged |
| `run_control_and_fields.py` | rc 0; `All checks passed`, until-decay 281 samples — control, unchanged |
| `cad_mesh_import_demo.py` | rc 0; plate realizes 2 cells, walls at z = 14 / 15 / 16 mm |
| `examples/config/microstrip_thru.yaml` | builds; sheets at z-node 12 / 14 = 2.0 / 3.0 mm |

## Not recomputed here, on purpose

* `tests/data/example_fidelity_snapshot.json` — the ingest phase regenerates it
  with `JAX_ENABLE_X64=0 python scripts/capture_example_fidelity_snapshot.py`
  AFTER the preflight and crossval groups land, because the snapshot covers
  validation scripts too. See `E-examples-example_fidelity_snapshot.md`.
* `docs/public/gallery/assets/**` and the committed `sparams.json` / `.s1p`
  from `scripts/_gallery_v3_patch_figs.py` and
  `scripts/precompute_gallery_artifacts.py` — outside this group's ownership.
  The gallery script's mesh changed from 1 mm to 0.5 mm cells (it had to: its
  foils were half-cell PEC Boxes, which the contract refuses), so its assets
  cost ~10x per case now.
* `scripts/patch_edgefed_s11_validation.py` — its own long-window run
  (`num_periods=200`) and the locks in
  `tests/locks/test_patch_edgefed_s11_passivity.py` /
  `test_patch_edgefed_resonance_harminv.py`. The board moved onto the lattice
  (dx 0.197 mm -> h_sub/4 = 0.19675 mm) and the metal moved onto the board
  faces, so both locks will move. Owned by whoever owns `tests/locks/`.
