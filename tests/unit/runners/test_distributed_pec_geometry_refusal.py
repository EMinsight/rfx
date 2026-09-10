"""The distributed lanes refuse every kind of declared PEC, not only sheets.

Both distributed runners assemble ``pec_mask`` and never apply it: their step
bodies call the DOMAIN-FACE PEC alone. That gap predates #931. What #931 added
was a refusal whose remedy told the user to redraw an unsupported sheet as a
VOLUME — advice that on these lanes produces a run with the metal still
missing and nothing to show for it. Measured before the fix: a two-device run
probing inside a declared PEC Box returned a trace bit-identical to the same
model with the Box deleted.

So the refusal now covers volumes too, and the remedy names what works.
"""
# Simulate 2 devices on CPU. Must be set BEFORE importing JAX.
import os  # noqa: I001

os.environ.setdefault(
    "XLA_FLAGS", "--xla_force_host_platform_device_count=2"
)

import warnings  # noqa: E402

import jax  # noqa: E402
import pytest  # noqa: E402

from rfx import Box, Simulation  # noqa: E402

DOMAIN = (24e-3, 12e-3, 12e-3)
DX = 1e-3


def _build(with_metal):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = Simulation(freq_max=15e9, domain=DOMAIN, dx=DX, boundary="pec")
        if with_metal:
            sim.add(Box((10e-3, 2e-3, 2e-3), (14e-3, 10e-3, 10e-3)),
                    material="pec")
        sim.add_source(position=(4e-3, 6e-3, 6e-3), component="ez",
                       amplitude_kind="field")
        sim.add_probe(position=(12e-3, 6e-3, 6e-3), component="ez")
    return sim


def _assert_remedy_is_honest(msg):
    assert "PEC volume" in msg, msg
    assert "does NOT help" in msg, (
        "the refusal must say that redrawing as a volume is not a remedy "
        "on this lane")
    assert "sim.run()" in msg, msg


def test_the_shmap_distributed_lane_refuses_a_declared_volume():
    if len(jax.devices()) < 2:
        pytest.skip("need 2 virtual devices "
                    "(XLA_FLAGS=--xla_force_host_platform_device_count=2)")
    from rfx.runners.distributed_v2 import run_distributed

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            run_distributed(_build(True), n_steps=4)
    _assert_remedy_is_honest(str(excinfo.value))

    # the control: without the conductor the same model runs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_distributed(_build(False), n_steps=4)
    assert res.time_series is not None


def test_the_pmap_distributed_lane_refuses_a_declared_volume():
    """The lane ``run_distributed_v2`` delegates to at one device, which
    carried a third copy of the same message."""
    from rfx.runners.distributed import run_distributed as pmap_run

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(NotImplementedError) as excinfo:
            pmap_run(_build(True), n_steps=4, devices=jax.devices()[:1])
    _assert_remedy_is_honest(str(excinfo.value))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = pmap_run(_build(False), n_steps=4, devices=jax.devices()[:1])
    assert res.time_series is not None
