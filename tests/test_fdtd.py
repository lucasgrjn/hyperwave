"""Basic tests for the ``fdtd.simulate()`` API."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from hyperwave import fdtd
from hyperwave.typing import Subvolume


def test_fdtd_simulation():
    snapshot_range = fdtd.SnapshotRange(
        start=10,
        interval=2,
        num=3,
    )
    num_steps = snapshot_range.start + snapshot_range.interval * (
        snapshot_range.num - 1
    )
    state, outs = fdtd.simulate(
        dt=1.0,
        grid=(jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2))),
        permittivity=jnp.ones((3, 1, 1, 1)),
        conductivity=jnp.zeros((3, 1, 1, 1)),
        source=fdtd.Source(
            offset=(0, 0, 0),
            field=-jnp.ones((3, 1, 1, 1)),
            waveform=jnp.ones((num_steps,)),
        ),
        output_volumes=[Subvolume((0, 0, 0), (2, 1, 1))],
        snapshot_range=snapshot_range,
    )
    assert state.e_field.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(
        outs[0][:, 0, 0, 0, 0],
        snapshot_range.start
        + snapshot_range.interval * jnp.arange(snapshot_range.num)
        + 1,
    )


# TODO: Test continuity. That is, that we can get the same result with two simulations as with a single simulation.
