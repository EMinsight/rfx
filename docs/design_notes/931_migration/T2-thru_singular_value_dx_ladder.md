# T2 → owner of `scripts/diagnostics/thru_singular_value_dx_ladder.py`

The three rung fixtures
(`tests/fixtures/thru_singular_value_dx_ladder/rung_dx_over_{1,2,4}.json`)
cannot be recaptured until this script declares its trace the way the contract
says. The board is already on-lattice — the trace height is `H_M = 1.0 mm` and
`dx` is 0.5 / 0.25 / 0.125 mm, so `H_M/dx` is 2 / 4 / 8 — so **only the
declaration changes**, not the mesh.

## What is wrong today

Line 91 and 101-105:

```python
def build_rung(divisor: int) -> tuple[Simulation, dict]:
    """The THRU at dx = DX0/divisor with the sheet trace and physical overhang."""
    ...
    # One-cell PEC sheet on top of the wire spans; overhang held at 0.5 mm.
    trace_lo = (X1_M - OVERHANG_M, Y_MID_M - W_M / 2, H_M)
    trace_hi = (X2_M + OVERHANG_M, Y_MID_M + W_M / 2, H_M + dx)
    sim.add(Box(trace_lo, trace_hi), material="pec")
```

The docstring and the comment both call it a sheet; the geometry is a
one-cell **volume**. Under §1.2 it realizes tangential walls at BOTH `H_M` and
`H_M + dx` with the Ez between them shorted — a 0.5 / 0.25 / 0.125 mm slab of
metal that thins with the mesh, which is precisely the mesh-dependent thickness
the contract exists to remove from a convergence ladder.

## Replacement

```python
    # 35 µm foil: a SHEET (#931 §1.3), declared by a zero-thickness Box on the
    # trace plane. H_M / dx is 2 / 4 / 8 across the ladder, so the plane is a
    # node line at every rung and the sheet lands on it exactly. Written as a
    # one-cell Box before the contract, it was a VOLUME whose realized
    # thickness followed the mesh — the ladder's own independent variable.
    trace_lo = (X1_M - OVERHANG_M, Y_MID_M - W_M / 2, H_M)
    trace_hi = (X2_M + OVERHANG_M, Y_MID_M + W_M / 2, H_M)
    sim.add(Box(trace_lo, trace_hi), material="pec")
```

## What the rung records must also carry

`tests/unit/sparams/test_thru_singular_value_dx_ladder_replay.py::test_g4_rasterization_scales_as_a_sheet`
reads `rasterization.finite_pec_cells == [340, 1360, 5440]` and asserts the
dx⁻² law on it. A sheet owns NO cell, so that key goes to zero on every rung
and stops carrying the property. The rasterization record needs the sheet's own
size instead — add, beside `finite_pec_cells`:

```python
    "sheet_footprint_nodes": <int>,   # nodes in the realized sheet footprint
    "sheet_planes": {"z": [<int>]},   # realized plane index per normal axis
```

both read from the single owner, e.g.

```python
    from tests._realized_geometry import realized      # or the equivalent
    rz = realized(sim)                                 # no solve
    fp = np.zeros(tuple(rz.grid.shape), dtype=bool)
    for sp in rz.sheets:
        fp |= np.asarray(sp.footprint, dtype=bool)
    rec["sheet_footprint_nodes"] = int(fp.sum())
    rec["sheet_planes"] = {"z": sorted(rz.sheet_planes.get(2, []))}
```

The footprint node count scales dx⁻² exactly as the cell count did, so G4 keeps
its law and states it on the quantity that still exists.

The wire-port half of G4 also moves: `live_flags[-1] is False` and
`n_live == 2*d` were true because the port's top cell landed inside the
one-cell PEC volume. Against a sheet the port's Ez is NORMAL to the conductor
and stays live by contract (§1.3), so the record will show `n_live == n_cells`
and all flags True. That is a realization change, not a regression — and it
means the two `wire_port_dead_extent_cells` entries in `BATTERY_CODES` will not
fire, so the expected preflight code list changes with the rungs.

## Recapture

Three runs, one per rung:

```
python scripts/diagnostics/thru_singular_value_dx_ladder.py \
    --dx-divisor {1,2,4} --output tests/fixtures/thru_singular_value_dx_ladder/rung_dx_over_{1,2,4}.json
```

The existing rung JSONs are frozen records of VESSL run 369367257803. They get
a NEW record, not a value edit — the same treatment the coax↔MSL
PREDECLARATION blocks get. `--dx-divisor 1` reproduces the battery fixture and
its recorded `sv_max = 1.003227`; that value is measured on the pre-contract
trace, so the note's gate G1 needs re-declaring against the new run rather than
being carried across.
