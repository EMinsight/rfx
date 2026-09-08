"""Resolve mesh inputs before either lane selection or grid construction.

The private mesh fields are resolved views; their dictionary slots retain the
declaration. This also covers readers outside Simulation (exporters, viewers,
optimizers), without an entry-point decorator or a list of callers to maintain.
"""
from __future__ import annotations

import warnings

import jax
import numpy as np


class AutoMeshWarning(UserWarning):
    """Mesh selection information, not a preflight legality finding."""


class _MeshField:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, sim, owner=None):
        if sim is None:
            return self
        return sim._resolve_mesh()[self.name]

    def __set__(self, sim, value):
        sim.__dict__[self.name] = value
        sim.__dict__.pop("_mesh_resolution", None)


class _MeshMixin:
    _dx = _MeshField()
    _domain = _MeshField()
    _dx_profile = _MeshField()
    _dy_profile = _MeshField()
    _dz_profile = _MeshField()

    def _resolve_mesh(self):
        """Return one cached, host-side resolution of the current declaration.

        Geometry/material records are immutable. Keep strong references and
        compare identity so traced arrays never enter equality/hash operations.
        Builder additions/replacements invalidate the view, including copies
        made by parameter sweeps. Explicit mesh inputs pass through unchanged.
        No resolved value is written back into the declared mesh slots.
        """
        state = self.__dict__
        names = ("_dx", "_domain", "_dx_profile", "_dy_profile", "_dz_profile")
        declared = {name: state.get(name) for name in names}
        geometry = state.get("_geometry", ())
        sheets = state.get("_thin_conductors", ())
        if declared["_dx"] is not None or not (geometry or sheets):
            return declared
        inputs = (
            *(state.get(name) for name in names),
            state["_freq_max"], state["_boundary"],
            *geometry, *sheets,
            *(item for pair in state["_materials"].items() for item in pair),
        )
        cached = state.get("_mesh_resolution")
        if cached is not None:
            previous, resolved = cached
            if len(inputs) == len(previous) and all(
                a is b for a, b in zip(inputs, previous)
            ):
                return resolved
        # Static geometry must be resolved on the host even when forward is
        # first called inside jit/grad. A traced geometry cannot define array
        # shapes; callers designing geometry must supply an explicit mesh.
        try:
            with jax.ensure_compile_time_eval():
                config = self._auto_configure_mesh()
        except (jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerArrayConversionError) as exc:
            raise ValueError(
                "Auto mesh requires static geometry and materials; supply "
                "dx= explicitly (and fixed profiles) before tracing design inputs."
            ) from exc
        resolved = dict(declared, _dx=config.dx)
        if config.dz_profile is not None and declared["_dz_profile"] is None:
            resolved["_dz_profile"] = config.dz_profile
            domain = declared["_domain"]
            resolved["_domain"] = (domain[0], domain[1], float(np.sum(config.dz_profile)))
        state["_mesh_resolution"] = (inputs, resolved)
        warnings.warn(
            f"Auto mesh: dx={config.dx * 1e3:.3f}mm "
            f"({config.cells_per_wavelength:.0f} cells/λ)"
            + (f", non-uniform z ({len(resolved['_dz_profile'])} cells)"
               if resolved["_dz_profile"] is not None else "")
            + ". Set dx= explicitly to suppress.",
            AutoMeshWarning, stacklevel=3,
        )
        for message in config.warnings:
            warnings.warn(message, AutoMeshWarning, stacklevel=3)
        return resolved

    @property
    def _uses_nonuniform_mesh(self):
        mesh = self._resolve_mesh()
        return any(mesh[name] is not None for name in (
            "_dx_profile", "_dy_profile", "_dz_profile"))

    def _build_realized_grid(self):
        """Build the selected grid for consumers supporting either lane."""
        return (self._build_nonuniform_grid() if self._uses_nonuniform_mesh
                else self._build_grid())

    def _require_uniform_mesh(self, consumer):
        if self._uses_nonuniform_mesh:
            raise NotImplementedError(
                f"{consumer} does not support the resolved non-uniform mesh. "
                "Use a non-uniform-capable entry point or supply an explicit "
                "uniform dx= and resolve all geometry thicknesses."
            )
