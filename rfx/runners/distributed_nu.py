"""Phase B: Non-uniform FDTD kernels for the shard_map distributed runner.

This module supplies the NU analogues of the uniform kernels used by
``rfx/runners/distributed_v2.py`` so the distributed path can accept
``NonUniformGrid`` without silently dropping the profile. The uniform
kernels in ``distributed.py`` remain the reference baseline and must
not be modified from this module.

Scope (Phase B minimal):
- x-axis shard only (1D slab decomposition).
- Global grading ratio <= 5:1 (shared single dt).
- x-axis CPML cells are uniform (guaranteed by make_nonuniform_grid
  boundary padding).
- TFSF single-device only (enforced upstream).
- Dispersive (Debye/Lorentz) E on NU distributed is NOT implemented
  here. The public entry point in ``distributed_v2`` falls back when
  dispersion is active.

Key helper: ``_build_sharded_inv_dx_arrays`` returns per-device
slabs of ``inv_dx`` / ``inv_dx_h`` whose slab boundary entry of
``inv_dx_h`` is derived from the global spacing straddling the slab
seam (NOT from the local slab alone) so H-field mean-spacing math
remains consistent across the shard boundary.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from rfx.core.yee import (
    FDTDState,
    MaterialArrays,
    MU_0,
    EPS_0,
    _shift_fwd,
    _shift_bwd,
)


# ---------------------------------------------------------------------------
# Sharded inv-spacing arrays
# ---------------------------------------------------------------------------

def _build_sharded_inv_dx_arrays(grid, n_devices, pad_x=0):
    """Build per-device x-axis inverse-spacing slabs for the shard_map runner.

    The caller has already padded the global x-extent by ``pad_x`` cells
    (to align ``nx`` on ``n_devices``).  We replicate that padding onto
    the cell-size profile using the boundary cell value (matching how
    ``make_nonuniform_grid`` pads CPML cells) and rebuild the global
    ``inv_dx`` and ``inv_dx_h`` from the padded profile, then reshape to
    per-device slabs.

    For ``inv_dx_h``, the last entry of each device's slab is the global
    mean-spacing straddling the slab seam with the next device (or 0 at
    the domain boundary), NOT derived from the local slab alone.

    Parameters
    ----------
    grid : NonUniformGrid
    n_devices : int
    pad_x : int
        Number of PEC-padded cells appended to the high-x end of the
        domain so that ``(nx + pad_x) % n_devices == 0``.

    Returns
    -------
    inv_dx_global : (nx_padded,) np.ndarray
        Replicated — every device sees the whole thing when used with
        ``P("x")`` (see caller packing).
    inv_dx_h_global : (nx_padded,) np.ndarray
    dx_padded : (nx_padded,) np.ndarray
        The padded cell-size profile (float32) — useful for diagnostics
        and the unit test.
    """
    dx_arr = np.asarray(grid.dx_arr, dtype=np.float64)
    if pad_x > 0:
        # pad at the high-x end with boundary-cell-size value
        dx_arr = np.concatenate(
            [dx_arr, np.full(pad_x, float(dx_arr[-1]))]
        )
    nx = dx_arr.shape[0]
    if nx % n_devices != 0:
        raise ValueError(
            f"After padding nx={nx} is not divisible by n_devices={n_devices}"
        )

    inv_dx = 1.0 / dx_arr
    # inv_dx_h[i] = 2 / (dx[i] + dx[i+1]) for i<N-1 ; 0 at end.
    inv_dx_h_mean = 2.0 / (dx_arr[:-1] + dx_arr[1:])
    inv_dx_h = np.concatenate([inv_dx_h_mean, np.zeros(1, dtype=np.float64)])

    return (
        inv_dx.astype(np.float32),
        inv_dx_h.astype(np.float32),
        dx_arr.astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Local NU update kernels (operate on per-device slab including ghosts)
# ---------------------------------------------------------------------------

def _update_h_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full,
                      inv_dx_h_slab, inv_dy_h_full, inv_dz_h_full):
    """H update on a local slab using NU inverse spacings.

    Mirrors ``rfx/core/yee.py::update_h_nu`` but accepts pre-sliced
    per-device ``inv_dx`` / ``inv_dx_h`` (length nx_local), while
    y/z spacings are replicated (full-axis).
    """
    ex, ey, ez = state.ex, state.ey, state.ez
    mu = materials.mu_r * MU_0

    curl_x = (
        (_shift_fwd(ez, 1) - ez) * inv_dy_h_full[None, :, None]
        - (_shift_fwd(ey, 2) - ey) * inv_dz_h_full[None, None, :]
    )
    curl_y = (
        (_shift_fwd(ex, 2) - ex) * inv_dz_h_full[None, None, :]
        - (_shift_fwd(ez, 0) - ez) * inv_dx_h_slab[:, None, None]
    )
    curl_z = (
        (_shift_fwd(ey, 0) - ey) * inv_dx_h_slab[:, None, None]
        - (_shift_fwd(ex, 1) - ex) * inv_dy_h_full[None, :, None]
    )

    hx = state.hx - (dt / mu) * curl_x
    hy = state.hy - (dt / mu) * curl_y
    hz = state.hz - (dt / mu) * curl_z

    return state._replace(hx=hx, hy=hy, hz=hz)


def _update_e_local_nu(state, materials, dt,
                      inv_dx_slab, inv_dy_full, inv_dz_full):
    """E update on a local slab using NU inverse (cell-local) spacings.

    Mirrors ``rfx/core/yee.py::update_e_nu``.
    """
    hx, hy, hz = state.hx, state.hy, state.hz
    eps = materials.eps_r * EPS_0
    sigma = materials.sigma

    sigma_dt_2eps = sigma * dt / (2.0 * eps)
    ca = (1.0 - sigma_dt_2eps) / (1.0 + sigma_dt_2eps)
    cb = (dt / eps) / (1.0 + sigma_dt_2eps)

    curl_x = (
        (hz - _shift_bwd(hz, 1)) * inv_dy_full[None, :, None]
        - (hy - _shift_bwd(hy, 2)) * inv_dz_full[None, None, :]
    )
    curl_y = (
        (hx - _shift_bwd(hx, 2)) * inv_dz_full[None, None, :]
        - (hz - _shift_bwd(hz, 0)) * inv_dx_slab[:, None, None]
    )
    curl_z = (
        (hy - _shift_bwd(hy, 0)) * inv_dx_slab[:, None, None]
        - (hx - _shift_bwd(hx, 1)) * inv_dy_full[None, :, None]
    )

    ex = ca * state.ex + cb * curl_x
    ey = ca * state.ey + cb * curl_y
    ez = ca * state.ez + cb * curl_z

    return state._replace(ex=ex, ey=ey, ez=ez, step=state.step + 1)


# ---------------------------------------------------------------------------
# Sharded NU grid metadata — Phase 2A
# ---------------------------------------------------------------------------

from typing import NamedTuple as _NamedTuple


class ShardedNUGrid(_NamedTuple):
    """Metadata describing a non-uniform grid that has been sliced into
    x-axis slabs for the shard_map distributed runner.

    **Coordinate mapping convention** (used by Phase 3 probe/source routing):

    Physical positions are always resolved on the *full-domain*
    ``NonUniformGrid`` first via ``position_to_index(grid, pos)`` which
    returns a global triple ``(i_global, j, k)``.  The x-index is then
    mapped to a rank and a local index:

        rank      = i_global // nx_per_rank
        local_i   = (i_global % nx_per_rank) + ghost_width

    where ``nx_per_rank = nx_local_real`` (the per-rank real cell count,
    *not* the padded/ghost count) and ``ghost_width`` is the ghost cell
    offset stored in this object.  No per-rank physical coordinate system
    is introduced — the global cumulative x-positions are the only
    reference frame.

    Fields
    ------
    nx : int
        Original (unpadded) global x cell count.
    ny : int
        Global y cell count (unchanged by sharding).
    nz : int
        Global z cell count (unchanged by sharding).
    n_devices : int
        Number of ranks / devices.
    nx_padded : int
        Global x count after PEC padding so ``nx_padded % n_devices == 0``.
    pad_x : int
        Number of PEC cells appended at the high-x end
        (``nx_padded - nx``).
    nx_per_rank : int
        Real cells per rank (``nx_padded // n_devices``).
    nx_local : int
        Per-rank cell count including ghost cells
        (``nx_per_rank + 2 * ghost_width``).
    ghost_width : int
        Number of ghost cells on each side of a rank's slab (always 1).
    cpml_layers : int
        CPML layer count from the source grid (replicated, same on every rank).
    dt : float
        Global shared timestep (same on every rank; not recomputed).
    inv_dx_global : np.ndarray  shape (nx_padded,)
        Cell-local inverse x-spacings for the padded full domain.
    inv_dx_h_global : np.ndarray  shape (nx_padded,)
        Mean-spacing inverse x-spacings (seam-aware) for the padded full domain.
    dx_padded : np.ndarray  shape (nx_padded,)
        Padded cell-size profile (float32); useful for diagnostics.
    inv_dy : np.ndarray  shape (ny,)
        Replicated y inverse spacings (every rank receives the full array).
    inv_dy_h : np.ndarray  shape (ny,)
        Replicated y mean-spacing inverse spacings.
    inv_dz : np.ndarray  shape (nz,)
        Replicated z inverse spacings.
    inv_dz_h : np.ndarray  shape (nz,)
        Replicated z mean-spacing inverse spacings.
    rank_has_high_x_pad : int
        Index of the rank that owns the high-x PEC padding cells
        (always ``n_devices - 1``; stored for Phase 3 trim logic).
    nx_trim : int
        Number of padded cells that must be trimmed from the high-x rank's
        slab when assembling the full-domain result (equals ``pad_x``).
    x_starts : tuple[int, ...]
        Global x start index (inclusive) of the real cells for each rank.
    x_stops : tuple[int, ...]
        Global x stop index (exclusive) of the real cells for each rank.
    """

    nx: int
    ny: int
    nz: int
    n_devices: int
    nx_padded: int
    pad_x: int
    nx_per_rank: int
    nx_local: int
    ghost_width: int
    cpml_layers: int
    dt: float
    inv_dx_global: object   # np.ndarray (nx_padded,) float32
    inv_dx_h_global: object  # np.ndarray (nx_padded,) float32
    dx_padded: object        # np.ndarray (nx_padded,) float32
    inv_dy: object           # np.ndarray (ny,) float32
    inv_dy_h: object         # np.ndarray (ny,) float32
    inv_dz: object           # np.ndarray (nz,) float32
    inv_dz_h: object         # np.ndarray (nz,) float32
    rank_has_high_x_pad: int
    nx_trim: int
    x_starts: tuple
    x_stops: tuple


def split_1d_with_ghost(arr: "np.ndarray", n_devices: int, nx_per: int,
                        nx_local: int, ghost: int,
                        pad_value: float) -> "np.ndarray":
    """Split a 1-D inverse-spacing array into per-device slabs with ghost cells.

    This is the canonical split helper shared between the NU metadata builder
    and the distributed_v2 runner.  It produces a ``(n_devices, nx_local)``
    NumPy array where each row is one rank's slab including ``ghost`` cells on
    each side.

    Parameters
    ----------
    arr : np.ndarray  shape (n_devices * nx_per,)
        Padded global inverse-spacing array (output of
        ``_build_sharded_inv_dx_arrays``).
    n_devices : int
    nx_per : int
        Real cells per device (``arr.shape[0] // n_devices``).
    nx_local : int
        ``nx_per + 2 * ghost``.
    ghost : int
        Ghost width (typically 1).
    pad_value : float
        Value to fill boundary ghost cells (1.0 for inv_dx, 0.0 for inv_dx_h).

    Returns
    -------
    slabs : np.ndarray  shape (n_devices, nx_local)
    """
    slabs = np.zeros((n_devices, nx_local), dtype=arr.dtype)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs[d, ghost:ghost + nx_per] = arr[lo:hi]
        # left ghost
        if d > 0:
            slabs[d, 0] = arr[lo - 1]
        else:
            slabs[d, 0] = pad_value
        # right ghost
        if d < n_devices - 1:
            slabs[d, -1] = arr[hi]
        else:
            slabs[d, -1] = pad_value
    return slabs


def build_sharded_nu_grid(
    grid,
    n_devices: int,
    exchange_interval: int = 1,
) -> ShardedNUGrid:
    """Build a :class:`ShardedNUGrid` from a full-domain :class:`NonUniformGrid`.

    This is the Phase 2A metadata-only helper.  It does **not** touch
    JAX device placement or shard_map; callers (e.g. the Phase 2B scan
    body) are responsible for calling ``jax.device_put`` on the returned
    arrays.

    Parameters
    ----------
    grid : NonUniformGrid
        Full-domain non-uniform grid produced by ``make_nonuniform_grid``.
    n_devices : int
        Number of ranks / devices for the x-slab decomposition.
    exchange_interval : int
        Ghost exchange interval.  Currently only ``exchange_interval == 1``
        is supported (one exchange per FDTD step).  The parameter is
        accepted for forward-compatibility with Phase 2E batched exchange.
        Ghost width is always ``1 * exchange_interval`` cells, so passing
        a larger value will increase ``ghost_width`` accordingly if support
        is added in a later phase.

    Returns
    -------
    ShardedNUGrid
        Immutable metadata object.  All numpy arrays are float32 and live
        on the host (CPU) at this stage.

    Notes
    -----
    **Coordinate mapping convention** (important for Phase 3):

    Probe and source physical positions must be converted to
    ``(i_global, j, k)`` using ``position_to_index(grid, pos)`` on the
    *full-domain* grid **before** sharding.  The resulting global ``i``
    is then mapped to a (rank, local_i) pair as::

        rank    = i_global // sharded.nx_per_rank
        local_i = (i_global % sharded.nx_per_rank) + sharded.ghost_width

    No per-rank physical coordinate system should be created.
    """
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 is reserved for Phase 2E; "
            "only exchange_interval=1 is supported in Phase 2A."
        )

    ghost = exchange_interval  # ghost_width = exchange_interval cells

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Pad nx to nearest multiple of n_devices (PEC cells on high-x end)
    pad_x = 0
    if nx % n_devices != 0:
        pad_x = n_devices - (nx % n_devices)
    nx_padded = nx + pad_x

    nx_per = nx_padded // n_devices
    nx_local = nx_per + 2 * ghost

    # Build padded inverse-spacing arrays (reuses existing Phase B helper)
    inv_dx_global, inv_dx_h_global, dx_padded = _build_sharded_inv_dx_arrays(
        grid, n_devices, pad_x=pad_x
    )

    # Replicate y/z inverse spacings (unchanged by x-sharding)
    inv_dy = np.asarray(grid.inv_dy, dtype=np.float32)
    inv_dy_h = np.asarray(grid.inv_dy_h, dtype=np.float32)
    inv_dz = np.asarray(grid.inv_dz, dtype=np.float32)
    inv_dz_h = np.asarray(grid.inv_dz_h, dtype=np.float32)

    # Rank x-range bookkeeping
    x_starts = tuple(d * nx_per for d in range(n_devices))
    x_stops = tuple(min((d + 1) * nx_per, nx) for d in range(n_devices))

    return ShardedNUGrid(
        nx=nx,
        ny=ny,
        nz=nz,
        n_devices=n_devices,
        nx_padded=nx_padded,
        pad_x=pad_x,
        nx_per_rank=nx_per,
        nx_local=nx_local,
        ghost_width=ghost,
        cpml_layers=grid.cpml_layers,
        dt=float(grid.dt),
        inv_dx_global=inv_dx_global,
        inv_dx_h_global=inv_dx_h_global,
        dx_padded=dx_padded,
        inv_dy=inv_dy,
        inv_dy_h=inv_dy_h,
        inv_dz=inv_dz,
        inv_dz_h=inv_dz_h,
        rank_has_high_x_pad=n_devices - 1,
        nx_trim=pad_x,
        x_starts=x_starts,
        x_stops=x_stops,
    )


# ---------------------------------------------------------------------------
# Phase 2B: hard-PEC + ghost-exchange sharded NU scan body
# ---------------------------------------------------------------------------

def shard_pec_mask_x_slab(global_mask, sharded_grid: ShardedNUGrid):
    """Slice a global PEC mask along x using the slab ownership of
    :class:`ShardedNUGrid`.

    The PEC mask is a boolean ``(nx, ny, nz)`` array.  Sharding it along
    the same x-slab partition as ``eps_r`` / ``sigma`` ensures that a
    PEC cell at global x-index ``i`` is owned by exactly one rank
    (``rank = i // nx_per_rank``) and not double-zeroed by the union of
    multiple ranks' ``apply_pec_mask`` calls.

    This helper handles ``pad_x`` padding (PEC=True for high-x padded
    cells) and the canonical split-with-ghost slabbing convention used
    by ``_split_state`` / ``_split_materials``.

    Parameters
    ----------
    global_mask : (nx, ny, nz) jnp.ndarray bool, or None
        Full-domain PEC mask.  Returns ``None`` if ``global_mask`` is
        ``None`` so callers can do an unconditional call.
    sharded_grid : ShardedNUGrid

    Returns
    -------
    sharded_mask : (n_devices * nx_local, ny, nz) jnp.ndarray bool, or None
        The mask reshaped to be x-sharded with ``P("x")``.  Each device
        sees ``(nx_local, ny, nz)``.
    """
    if global_mask is None:
        return None

    n_devices = sharded_grid.n_devices
    nx_per = sharded_grid.nx_per_rank
    nx_local = sharded_grid.nx_local
    ghost = sharded_grid.ghost_width
    pad_x = sharded_grid.pad_x
    ny = sharded_grid.ny
    nz = sharded_grid.nz

    # Pad along x with PEC=True (consistent with high-x PEC padding)
    if pad_x > 0:
        pad_widths = [(0, pad_x), (0, 0), (0, 0)]
        global_mask = jnp.pad(global_mask, pad_widths, constant_values=True)

    nx_padded = global_mask.shape[0]
    assert nx_padded == n_devices * nx_per, (
        f"sharded_pec_mask: padded mask shape {nx_padded} != "
        f"n_devices*nx_per_rank = {n_devices * nx_per}"
    )

    # Build per-device slabs with ghost cells.  Ghost cells at the
    # physical boundary are PEC=True (matches the high-x pad and is
    # consistent with apply_pec on the domain face); interior ghosts
    # carry the neighbour's PEC status so apply_pec_mask sees a
    # correct neighbour-set when computing tangential masks.
    slabs = jnp.zeros((n_devices, nx_local, ny, nz), dtype=jnp.bool_)
    for d in range(n_devices):
        lo = d * nx_per
        hi = lo + nx_per
        slabs = slabs.at[d, ghost:ghost + nx_per, :, :].set(global_mask[lo:hi])
        if d > 0:
            slabs = slabs.at[d, 0, :, :].set(global_mask[lo - 1])
        else:
            # Domain boundary at x_lo: ghost is PEC=True (matches apply_pec)
            slabs = slabs.at[d, 0, :, :].set(True)
        if d < n_devices - 1:
            slabs = slabs.at[d, -1, :, :].set(global_mask[hi])
        else:
            # Domain boundary at x_hi: ghost is PEC=True
            slabs = slabs.at[d, -1, :, :].set(True)

    # Reshape to sharded layout: (n_devices * nx_local, ny, nz)
    return slabs.reshape(n_devices * nx_local, ny, nz)


def _exchange_component_nu_shmap(field, mesh, n_devices):
    """Ghost-exchange one field component on an x-sharded array.

    Mirrors ``rfx/runners/distributed_v2.py::_exchange_component_shmap``
    exactly — the NU scan body uses the same ghost-exchange contract
    (last-real-cell -> next rank's left ghost, first-real-cell ->
    previous rank's right ghost) as the uniform distributed runner.
    """
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=P("x"),
        out_specs=P("x"),
        check_rep=False,
    )
    def _exchange(f):
        right_boundary = f[-2:-1, :, :]   # last real cell -> right neighbour's left ghost
        left_boundary = f[1:2, :, :]      # first real cell -> left neighbour's right ghost

        perm_right = [(i, (i + 1) % n_devices) for i in range(n_devices)]
        left_ghost_recv = lax.ppermute(right_boundary, "x", perm=perm_right)

        perm_left = [(i, (i - 1) % n_devices) for i in range(n_devices)]
        right_ghost_recv = lax.ppermute(left_boundary, "x", perm=perm_left)

        device_idx = lax.axis_index("x")

        left_ghost_val = jnp.where(device_idx > 0,
                                   left_ghost_recv,
                                   f[0:1, :, :])
        right_ghost_val = jnp.where(device_idx < n_devices - 1,
                                    right_ghost_recv,
                                    f[-1:, :, :])

        f = f.at[0:1, :, :].set(left_ghost_val)
        f = f.at[-1:, :, :].set(right_ghost_val)
        return f

    return _exchange(field)


def _exchange_h_ghosts_nu(state: FDTDState, mesh, n_devices: int) -> FDTDState:
    return state._replace(
        hx=_exchange_component_nu_shmap(state.hx, mesh, n_devices),
        hy=_exchange_component_nu_shmap(state.hy, mesh, n_devices),
        hz=_exchange_component_nu_shmap(state.hz, mesh, n_devices),
    )


def _exchange_e_ghosts_nu(state: FDTDState, mesh, n_devices: int) -> FDTDState:
    return state._replace(
        ex=_exchange_component_nu_shmap(state.ex, mesh, n_devices),
        ey=_exchange_component_nu_shmap(state.ey, mesh, n_devices),
        ez=_exchange_component_nu_shmap(state.ez, mesh, n_devices),
    )


def _apply_pec_face_nu_shmap(state: FDTDState, mesh, n_devices: int,
                             nx_local: int) -> FDTDState:
    """Apply PEC on physical domain faces (x_lo, x_hi, y, z) using shard_map.

    Mirrors ``rfx/runners/distributed_v2.py::_apply_pec_shmap`` exactly:
    Y- and Z-face PEC is local to every rank; X-face PEC is rank-
    conditional (only rank 0 zeroes x_lo, only rank N-1 zeroes x_hi).

    Critically, the X-face PEC acts on the **first real cell**
    (``ghost``) and the **last real cell** (``nx_local - 1 - ghost``),
    NOT on the seam ghost cells (which belong to neighbouring ranks).
    This is V3 bullet 7 ("Hard PEC only acts on physical boundary or
    masked cells, not on seam ghosts") and V3 bullet 6 ("Hard PEC does
    not re-zero interior seam cells of neighbouring ranks").
    """
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec(ex, ey, ez):
        ghost = 1

        # Y-axis PEC (every rank — the y boundary is local to all ranks)
        ex = ex.at[:, 0, :].set(0.0)
        ex = ex.at[:, -1, :].set(0.0)
        ez = ez.at[:, 0, :].set(0.0)
        ez = ez.at[:, -1, :].set(0.0)

        # Z-axis PEC (every rank — z boundary is local to all ranks)
        ex = ex.at[:, :, 0].set(0.0)
        ex = ex.at[:, :, -1].set(0.0)
        ey = ey.at[:, :, 0].set(0.0)
        ey = ey.at[:, :, -1].set(0.0)

        device_idx = lax.axis_index("x")

        # X-lo PEC: only rank 0; act on first REAL cell (skip ghost)
        is_first = (device_idx == 0)
        ey_xlo = jnp.where(is_first, 0.0, ey[ghost, :, :])
        ez_xlo = jnp.where(is_first, 0.0, ez[ghost, :, :])
        ey = ey.at[ghost, :, :].set(ey_xlo)
        ez = ez.at[ghost, :, :].set(ez_xlo)

        # X-hi PEC: only rank N-1; act on last REAL cell (skip ghost)
        is_last = (device_idx == n_devices - 1)
        last_real = nx_local - 1 - ghost
        ey_xhi = jnp.where(is_last, 0.0, ey[last_real, :, :])
        ez_xhi = jnp.where(is_last, 0.0, ez[last_real, :, :])
        ey = ey.at[last_real, :, :].set(ey_xhi)
        ez = ez.at[last_real, :, :].set(ez_xhi)

        return ex, ey, ez

    ex, ey, ez = _pec(state.ex, state.ey, state.ez)
    return state._replace(ex=ex, ey=ey, ez=ez)


def _apply_pec_mask_nu_shmap(state: FDTDState, sharded_pec_mask, mesh,
                             n_devices: int, nx_local: int) -> FDTDState:
    """Apply geometry-defined PEC mask zeroing on x-sharded fields.

    Each rank owns the PEC cells inside its real-cell range
    ``[ghost, ghost + nx_per_rank)``.  Per V3 bullet 6, we must NOT
    re-zero PEC cells that live in another rank's slab; per V3 bullet 7,
    seam ghost cells must not be acted on.

    The implementation:
      * computes the per-component tangential mask (PEC AND has-PEC-
        neighbour-in-tangential-axis) on the local slab including ghost
        cells, identical math to ``rfx/boundaries/pec.py::apply_pec_mask``;
      * gates the mask so ghost-cell rows are forced to ``False`` before
        zeroing the field — interior real cells use their slab-local
        neighbour computation, and the **first/last real cells** see the
        ghost neighbour (which carries the seam-neighbour's PEC status
        because ``shard_pec_mask_x_slab`` populated it).
    """
    if sharded_pec_mask is None:
        return state

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("x"), P("x"), P("x"), P("x")),
        out_specs=(P("x"), P("x"), P("x")),
        check_rep=False,
    )
    def _pec_mask(ex, ey, ez, mask):
        # Tangential masks (per-component): PEC AND has-PEC-neighbour
        # in the component's own direction.
        # Use _shift_fwd / _shift_bwd-like inline rolls without wrap so
        # ghost cells do not introduce wrap artefacts.  jnp.roll wraps
        # in pure JAX, but inside a slab that is fine because the slab
        # already includes the seam neighbours via ghost cells; ghost
        # rows are forced to False below.
        mask_ex = mask & (jnp.roll(mask, 1, axis=0) | jnp.roll(mask, -1, axis=0))
        mask_ey = mask & (jnp.roll(mask, 1, axis=1) | jnp.roll(mask, -1, axis=1))
        mask_ez = mask & (jnp.roll(mask, 1, axis=2) | jnp.roll(mask, -1, axis=2))

        # Force ghost rows to False so we never touch a neighbour rank's
        # cells.  Real cells span [ghost, nx_local - ghost).
        ghost = 1
        ghost_zero = jnp.zeros_like(mask_ex[0:1, :, :])
        mask_ex = mask_ex.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ex = mask_ex.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])
        mask_ey = mask_ey.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ey = mask_ey.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])
        mask_ez = mask_ez.at[0:ghost, :, :].set(ghost_zero[0:ghost, :, :])
        mask_ez = mask_ez.at[nx_local - ghost:nx_local, :, :].set(
            ghost_zero[0:ghost, :, :])

        ex = ex * (1.0 - mask_ex.astype(ex.dtype))
        ey = ey * (1.0 - mask_ey.astype(ey.dtype))
        ez = ez * (1.0 - mask_ez.astype(ez.dtype))
        return ex, ey, ez

    ex, ey, ez = _pec_mask(state.ex, state.ey, state.ez, sharded_pec_mask)
    return state._replace(ex=ex, ey=ey, ez=ez)


def run_nonuniform_distributed_pec(
    sharded_grid: ShardedNUGrid,
    sharded_materials: MaterialArrays,
    sharded_pec_mask,
    n_steps: int,
    sources: list = None,
    probes: list = None,
    *,
    n_devices: int,
    exchange_interval: int = 1,
    debye=None,
    lorentz=None,
    devices=None,
) -> dict:
    """Phase 2B sharded NU scan body — hard PEC + ghost exchange only.

    This runner implements the V3 Phase 2B contract (bullets 1-10 from
    ``docs/research_notes/2026-04-16_issue44_v3_plan.md`` lines 621-632):
    H/E NU updates on x-slabs with ghost-cell exchange at slab seams,
    domain-face PEC, and geometry/override-union PEC mask zeroing.

    Per-rank scan body ordering (mirrors single-device
    ``rfx/nonuniform.py::run_nonuniform``)::

        1. H update (update_h_nu)                    via shard_map
        2. Ghost exchange of H                       via lax.ppermute
        3. E update (update_e_nu) using exchanged H  via shard_map
        4. Source injection (rank-conditional)       via shard_map
        5. Ghost exchange of E                       via lax.ppermute
        6. apply_pec on physical domain faces        via shard_map
        7. apply_pec_mask (geometry + override)      via shard_map
        8. # Phase 2C splice point: apply_cpml_e here
        9. Probe accumulation (rank-conditional sum) via lax.psum

    Parameters
    ----------
    sharded_grid : ShardedNUGrid
        Output of :func:`build_sharded_nu_grid`.
    sharded_materials : MaterialArrays
        x-slab sharded ``(eps_r, sigma, mu_r)``; each component already
        placed on the mesh with ``P("x")`` sharding.  Layout:
        ``(n_devices * nx_local, ny, nz)``.
    sharded_pec_mask : jnp.ndarray bool or None
        x-slab sharded PEC mask, same layout as a material array, or
        ``None`` to skip mask zeroing.  Use :func:`shard_pec_mask_x_slab`.
    n_steps : int
        Total FDTD steps.
    sources : list of SourceSpec, optional
        Each ``SourceSpec.i`` is a global x-index (full-domain).  Routing
        to per-rank handlers uses
        ``rank = i // nx_per_rank,
         local_i = (i % nx_per_rank) + ghost_width``.
    probes : list of ProbeSpec, optional
        Same routing as sources.
    n_devices : int
        Required.  Must match ``sharded_grid.n_devices``.
    exchange_interval : int, optional
        Reserved for Phase 2E batched exchange; only ``1`` is supported.
    debye, lorentz : reserved
        Phase 2D dispatches dispersion via this entry point; passing
        non-None raises ``NotImplementedError`` ("Phase 2D pending").
    devices : list of jax.Device, optional
        If ``None``, uses ``jax.devices()[:n_devices]``.

    Returns
    -------
    dict
        ``{"time_series": (n_steps, n_probes) ndarray,
           "final_state":  FDTDState (gathered to full-domain),
           "final_state_sharded": FDTDState (sharded, in-mesh layout)}``
    """
    if n_devices != sharded_grid.n_devices:
        raise ValueError(
            f"n_devices={n_devices} != sharded_grid.n_devices="
            f"{sharded_grid.n_devices}"
        )
    if exchange_interval != 1:
        raise NotImplementedError(
            "exchange_interval > 1 reserved for Phase 2E; only 1 supported."
        )
    if debye is not None or lorentz is not None:
        raise NotImplementedError(
            "Phase 2D pending: dispersive (Debye/Lorentz) materials are "
            "not yet supported on the distributed NU PEC scan body."
        )

    sources = list(sources) if sources is not None else []
    probes = list(probes) if probes is not None else []

    # Defer the `Mesh` import to runtime so module import remains
    # lightweight (matches distributed_v2.py's pattern).
    from jax.sharding import Mesh, NamedSharding

    if devices is None:
        devices = jax.devices()[:n_devices]
    if len(devices) < n_devices:
        raise ValueError(
            f"Only {len(devices)} JAX devices available; need {n_devices}."
        )
    mesh = Mesh(np.array(devices), axis_names=("x",))
    shd = NamedSharding(mesh, P("x"))
    rep = NamedSharding(mesh, P())

    nx_local = sharded_grid.nx_local
    nx_per = sharded_grid.nx_per_rank
    ghost = sharded_grid.ghost_width
    ny = sharded_grid.ny
    nz = sharded_grid.nz
    nx_padded = sharded_grid.nx_padded
    pad_x = sharded_grid.pad_x
    dt = sharded_grid.dt

    # ------------------------------------------------------------------
    # Sharded inverse-spacing arrays (1-D)
    # ------------------------------------------------------------------
    inv_dx_slabs = split_1d_with_ghost(
        sharded_grid.inv_dx_global, n_devices, nx_per, nx_local, ghost,
        pad_value=1.0,
    )
    inv_dx_h_slabs = split_1d_with_ghost(
        sharded_grid.inv_dx_h_global, n_devices, nx_per, nx_local, ghost,
        pad_value=0.0,
    )
    inv_dx_sharded = jax.device_put(
        inv_dx_slabs.reshape(n_devices * nx_local), shd)
    inv_dx_h_sharded = jax.device_put(
        inv_dx_h_slabs.reshape(n_devices * nx_local), shd)
    inv_dy_rep = jax.device_put(sharded_grid.inv_dy, rep)
    inv_dy_h_rep = jax.device_put(sharded_grid.inv_dy_h, rep)
    inv_dz_rep = jax.device_put(sharded_grid.inv_dz, rep)
    inv_dz_h_rep = jax.device_put(sharded_grid.inv_dz_h, rep)

    # ------------------------------------------------------------------
    # Initial state — full padded domain, then slab-split + shard
    # ------------------------------------------------------------------
    from rfx.core.yee import init_state
    from rfx.runners.distributed import _split_state

    full_state = init_state((nx_padded, ny, nz))
    state_slabs = _split_state(full_state, n_devices, ghost)

    def _shard_stacked(arr):
        n_dev = arr.shape[0]
        rest = arr.shape[1:]
        return jax.device_put(arr.reshape(n_dev * rest[0], *rest[1:]), shd)

    sharded_state = FDTDState(
        ex=_shard_stacked(state_slabs.ex),
        ey=_shard_stacked(state_slabs.ey),
        ez=_shard_stacked(state_slabs.ez),
        hx=_shard_stacked(state_slabs.hx),
        hy=_shard_stacked(state_slabs.hy),
        hz=_shard_stacked(state_slabs.hz),
        step=jax.device_put(jnp.int32(0), rep),
    )

    # ------------------------------------------------------------------
    # Source / probe routing — V3 bullet (Phase 2A coordinate convention)
    # ------------------------------------------------------------------
    src_device_ids = []
    src_local_specs = []
    for s in sources:
        dev_id = s.i // nx_per
        local_i = (s.i % nx_per) + ghost
        src_device_ids.append(int(dev_id))
        src_local_specs.append((int(local_i), int(s.j), int(s.k), s.component))

    prb_device_ids = []
    prb_local_specs = []
    for p in probes:
        dev_id = p.i // nx_per
        local_i = (p.i % nx_per) + ghost
        prb_device_ids.append(int(dev_id))
        prb_local_specs.append((int(local_i), int(p.j), int(p.k), p.component))

    n_src = len(sources)
    n_prb = len(probes)

    if n_src > 0:
        src_waveforms = jnp.stack([s.waveform for s in sources], axis=-1)
    else:
        src_waveforms = jnp.zeros((n_steps, 0), dtype=jnp.float32)
    src_waveforms_rep = jax.device_put(src_waveforms, rep)

    # ------------------------------------------------------------------
    # Per-step shmap-wrapped helpers
    # ------------------------------------------------------------------
    def _update_h_shmap(st, mat):
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),  # ex, ey, ez
                P("x"), P("x"), P("x"),  # hx, hy, hz
                P(),                     # step
                P("x"), P("x"), P("x"),  # eps_r, sigma, mu_r
                P("x"), P(None), P(None),  # inv_dx, inv_dy, inv_dz
                P("x"), P(None), P(None),  # inv_dx_h, inv_dy_h, inv_dz_h
            ),
            out_specs=(P("x"), P("x"), P("x"), P()),
            check_rep=False,
        )
        def _h(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
               invdx, invdy, invdz, invdxh, invdyh, invdzh):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            new_st = _update_h_local_nu(
                _st, _mat, dt, invdx, invdy, invdz, invdxh, invdyh, invdzh)
            return new_st.hx, new_st.hy, new_st.hz, new_st.step

        hx, hy, hz, step = _h(
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
            inv_dx_h_sharded, inv_dy_h_rep, inv_dz_h_rep,
        )
        return st._replace(hx=hx, hy=hy, hz=hz, step=step)

    def _update_e_shmap(st, mat):
        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(
                P("x"), P("x"), P("x"),
                P("x"), P("x"), P("x"),
                P(),
                P("x"), P("x"), P("x"),
                P("x"), P(None), P(None),
            ),
            out_specs=(P("x"), P("x"), P("x"), P()),
            check_rep=False,
        )
        def _e(ex, ey, ez, hx, hy, hz, step, eps_r, sigma, mu_r,
               invdx, invdy, invdz):
            _st = FDTDState(ex=ex, ey=ey, ez=ez, hx=hx, hy=hy, hz=hz, step=step)
            _mat = MaterialArrays(eps_r=eps_r, sigma=sigma, mu_r=mu_r)
            new_st = _update_e_local_nu(_st, _mat, dt, invdx, invdy, invdz)
            return new_st.ex, new_st.ey, new_st.ez, new_st.step

        ex, ey, ez, step = _e(
            st.ex, st.ey, st.ez, st.hx, st.hy, st.hz, st.step,
            mat.eps_r, mat.sigma, mat.mu_r,
            inv_dx_sharded, inv_dy_rep, inv_dz_rep,
        )
        return st._replace(ex=ex, ey=ey, ez=ez, step=step)

    def _inject_sources_shmap(st, src_vals_step):
        if n_src == 0:
            return st

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"), P()),
            out_specs=(P("x"), P("x"), P("x")),
            check_rep=False,
        )
        def _inject(ex, ey, ez, sv):
            device_idx = lax.axis_index("x")
            for idx_s in range(n_src):
                li, lj, lk, lc = src_local_specs[idx_s]
                dev_id = src_device_ids[idx_s]
                val = jnp.where(device_idx == dev_id, sv[idx_s], 0.0)
                if lc == "ex":
                    ex = ex.at[li, lj, lk].add(val)
                elif lc == "ey":
                    ey = ey.at[li, lj, lk].add(val)
                elif lc == "ez":
                    ez = ez.at[li, lj, lk].add(val)
            return ex, ey, ez

        ex, ey, ez = _inject(st.ex, st.ey, st.ez, src_vals_step)
        return st._replace(ex=ex, ey=ey, ez=ez)

    def _sample_probes_shmap(st):
        if n_prb == 0:
            return jnp.zeros(0, dtype=jnp.float32)

        @partial(
            shard_map,
            mesh=mesh,
            in_specs=(P("x"), P("x"), P("x"),
                      P("x"), P("x"), P("x")),
            out_specs=P(),
            check_rep=False,
        )
        def _sample(ex, ey, ez, hx, hy, hz):
            device_idx = lax.axis_index("x")
            samples = []
            for idx_p in range(n_prb):
                li, lj, lk, lc = prb_local_specs[idx_p]
                dev_id = prb_device_ids[idx_p]
                if lc == "ex":
                    raw = ex[li, lj, lk]
                elif lc == "ey":
                    raw = ey[li, lj, lk]
                elif lc == "ez":
                    raw = ez[li, lj, lk]
                elif lc == "hx":
                    raw = hx[li, lj, lk]
                elif lc == "hy":
                    raw = hy[li, lj, lk]
                else:
                    raw = hz[li, lj, lk]
                val = jnp.where(device_idx == dev_id, raw, 0.0)
                samples.append(val)
            return lax.psum(jnp.stack(samples), "x")

        return _sample(st.ex, st.ey, st.ez, st.hx, st.hy, st.hz)

    # ------------------------------------------------------------------
    # Per-step scan body (Phase 2B ordering — see docstring)
    # ------------------------------------------------------------------
    def step_fn(carry, xs):
        _step_idx, src_vals = xs
        st = carry["fdtd"]

        # 1. H update (NU)
        st = _update_h_shmap(st, sharded_materials)

        # 2. Ghost exchange of H so the upcoming E update sees the
        #    neighbour rank's H at the seam.
        st = _exchange_h_ghosts_nu(st, mesh, n_devices)

        # 3. E update (NU) using exchanged H
        st = _update_e_shmap(st, sharded_materials)

        # 4. Source injection (rank-conditional via shard_map)
        st = _inject_sources_shmap(st, src_vals)

        # 5. Ghost exchange of E so the next step's H update sees the
        #    neighbour rank's E at the seam.
        st = _exchange_e_ghosts_nu(st, mesh, n_devices)

        # 6. PEC on physical domain faces (X-faces are rank-conditional).
        st = _apply_pec_face_nu_shmap(st, mesh, n_devices, nx_local)

        # 7. PEC mask zeroing (geometry + override union).  No-op when
        #    sharded_pec_mask is None.
        if sharded_pec_mask is not None:
            st = _apply_pec_mask_nu_shmap(
                st, sharded_pec_mask, mesh, n_devices, nx_local)

        # 8. Phase 2C splice point: apply_cpml_e here
        # ------------------------------------------------------------------
        # Phase 2C will hook the CPML E correction in at this point.
        # The contract is:
        #   * x-face CPML state must be rank-conditional (rank 0 owns
        #     x_lo, rank N-1 owns x_hi, interior ranks no-op).
        #   * y/z-face CPML state is sliced across local x extent on
        #     every rank.
        #   * Internal slab seams are NOT physical CPML boundaries.
        # See V3 plan lines 634-657 for the full Phase 2C contract.
        # ------------------------------------------------------------------

        # 9. Probe accumulation (rank-conditional sample + lax.psum)
        probe_out = _sample_probes_shmap(st)

        return {"fdtd": st}, probe_out

    # ------------------------------------------------------------------
    # JIT-compiled scan over n_steps
    # ------------------------------------------------------------------
    step_indices = jnp.arange(n_steps, dtype=jnp.int32)
    xs = (step_indices, src_waveforms_rep)

    carry_init = {"fdtd": sharded_state}
    run_fn = jax.jit(lambda c, xx: lax.scan(step_fn, c, xx))
    final_carry, probe_ts = run_fn(carry_init, xs)
    final_state_sharded = final_carry["fdtd"]

    # ------------------------------------------------------------------
    # Gather sharded state -> full-domain (nx, ny, nz)
    # ------------------------------------------------------------------
    from rfx.runners.distributed import gather_array_x

    def _unstack_and_gather(sharded_arr):
        arr = np.array(sharded_arr)
        total_x = arr.shape[0]
        assert total_x == n_devices * nx_local, (
            f"unstack: total_x={total_x} != n_devices*nx_local={n_devices * nx_local}"
        )
        stacked = arr.reshape(n_devices, nx_local, *arr.shape[1:])
        gathered = jnp.array(gather_array_x(jnp.array(stacked), ghost))
        if pad_x > 0:
            gathered = gathered[: sharded_grid.nx]
        return gathered

    final_state = FDTDState(
        ex=_unstack_and_gather(final_state_sharded.ex),
        ey=_unstack_and_gather(final_state_sharded.ey),
        ez=_unstack_and_gather(final_state_sharded.ez),
        hx=_unstack_and_gather(final_state_sharded.hx),
        hy=_unstack_and_gather(final_state_sharded.hy),
        hz=_unstack_and_gather(final_state_sharded.hz),
        step=jnp.int32(int(final_state_sharded.step)),
    )

    if n_prb > 0:
        time_series = jnp.array(probe_ts)
    else:
        time_series = jnp.zeros((n_steps, 0), dtype=jnp.float32)

    return {
        "time_series": time_series,
        "final_state": final_state,
        "final_state_sharded": final_state_sharded,
    }
