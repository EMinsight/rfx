# Retire the short as an obligatory advisory trigger

Decision, 2026-09-09. This supersedes only T11's short disposition. The PI
explicitly authorized changing its acceptance contract while preserving #931,
production code, and the separate MSL item.

## Property and derivation

The old witness protects **warning-policy integrity**: callers must see an
advisory for extracted outgoing power in the policy's soft range, with no
exception or mutation of the result; excessive/nonfinite results retain hard
handling. It does not protect a law requiring a passive short to create power.
For unit incident wave a_j=1, outgoing power is P_j=sum_i |S_ij|^2.
The published policy (CHANGELOG.md, "waveguide S-matrix soft over-unity
advisory", introduced at e7fb2ba774064cdcc4024df82578d5b5b07c91cb) is:

| Path | Silent column-power gate | Warn-only | Hard diagnostic (raise if strict) |
|---|---|---|---|
| normalize=False | P <= 2.25 | 2.25 < P <= 1+2 | P > 1+2 |
| normalize=True or flux | P <= 1+0.10 | empty soft interval | P > 1+0.10 |

Other diagnostics may independently warn. The tests distinguish the column
power advisory from reciprocity and per-entry amplitude warnings.

**Verdict:** `(2.25, 3]` remains the existing software advisory policy. It was
never derived as a physical short prediction. The 2.25 floor was chosen with
margin over a measured ~2.0 envelope, and the ~2.51 coarse realization was
chosen to exercise it. That realization double-counted one cavity as two
outgoing ports. Requiring a valid short to stay in this range fitted an invalid
measurement. We retire that physical requirement, without shifting or widening
any interval and without choosing a new numerical pin.

The replacement public-API test injects S at the numerical extractor boundary,
then exercises real assembly, normalization dispatch, result construction and
the diagnostic epilogue. Unit incident power and real outgoing entries give
P=1, 2.25, 2.5, 3, 3.25; exact float64 sums are checked before use. This covers
the open lower and closed upper boundaries, both sides, all three normalization
modes, strict/non-strict handling, and exact preservation of S. The helper tests
also check the adjacent floating-point values around |S|=1.5. These are policy
inputs derived from the boundary, not fitted FDTD outputs.

The repaired live fixture now protects **PEC isolation with observable drives**.
A full-cross-section PEC plate gives two uncoupled regions of the Maxwell update.
Zero initial fields and no source in the opposite region imply its V/I histories
are exactly zero, hence S12=S21=0 at every bin. Both driven a/b spectra and all
four local V/I histories must be finite and nonzero: an unobserved drive cannot
pass by returning an all-zero matrix. Existing compiled-plane geometry checks
protect both walls and the source/reference/probe connected regions. These
criteria require no fitted amplitude tolerance, settling or mesh extrapolation.
Coarse and fine are independent topology checks, not a mesh-convergence pair.

Reflection-accuracy coverage remains the separate
`tests/oracle/test_waveguide_port_validation_battery.py::test_pec_short_s11_magnitude`
(min >=0.99, max <1.03, mean within 0.02 of unity). Its committed record names
VESSL 369367259278, module 9 passed. That is historical coverage, not a fresh
run here and not qualification of this coarse fixture.

## Evidence and prelaunch audit

Relevant project memory directories are absent in this worktree; no other
worktree was consulted. The existing durable design notes supply the evidence:

- T11: "Changing that acceptance contract or starting a new absolute-reflection
  study is a separate decision". This decision is now explicitly authorized;
  it does not claim the old qualification succeeded.
- `fixture-repair-evidence/short-live-adjudication.md`: "Both transmission terms
  and all opposite-side traces are zero, as required for the closed
  full-cross-section plate." The new topology property is consistent with this
  independent observation and #931's full PEC wall ownership.
- #931 normative design: "a Box drawn ... on node planes realizes tangential
  walls at BOTH" faces. No product operator or alternate realization is changed.

Prior named run 369367259618 at b36fc46cdf21d1c57f221e6a057654bcad60bae2:
**15 passed, 1 failed**, coarse maximum **1.044479250907898**, fine
**1.0494229793548584**. Historical node sampling under current edge ownership
restored **2.527903795**, versus **2.527910233** with historical sigma damping;
at 6 GHz |S11|=1.133861328 and spurious |S21|=1.114568210. The controlled repair
preserved left S11 and its V/I traces bit-for-bit while removing false remote
power. Full committed captures and inspection figure are linked from T11.

Prelaunch checks: local policy/geometry **55 passed, 0 failed**; exact coarse
and fine node selections rehearsed separately with `-o addopts=` and timeout
settings. Cheap falsifiers inject a missing guard, wrong tolerance and a soft
advisory promoted to exception: all must be rejected by the public API gate.
R2: zero new physics attempts before this campaign; one predeclared topology
qualification per existing resolution. No parameter search or repeat of the
old causal interventions. Falsifier: any nonzero cross-region history/S entry,
missing driven wave, or wrong public diagnostic outcome fails qualification.

VESSL: use maintained base-pod/remilab-c0 from fixture-repair-short.yaml;
resource reference 369367259623 verifies 8 CPU/32 GiB. Status snapshots and
lab-status are unavailable here, so the maintained-definition fallback applies.
The job verifies cgroup minima, CPU backend, exact SHA and local import. Each
job gets a unique node-local clone and an independent artifact pointer. The
submitter copies YAML outside git and records the run ID. One pytest per pod,
policy -n 4; each physics job -n 0; all use thread timeouts. Prior short run
369367259618 logs are backed up and retained as claims-bearing evidence; other
campaigns and the running MSL refinement are outside this task's cleanup scope.

## Live result

Pending the three predeclared lanes; append exact runs, counts and traces before
closing. This preparation commit is not the final qualification claim.
