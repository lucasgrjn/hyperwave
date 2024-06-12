"""Basic tests for the ``fdtd.simulate()`` API."""

import jax.numpy as jnp
import numpy as np

from hyperwave import fdtd


def test_fdtd_simulation():
    output_spec = fdtd.OutputSpec(
        start=10,
        interval=2,
        num=3,
        offsets=[(0, 0, 0)],
        shapes=[(1, 1, 1)],
    )
    num_steps = output_spec.start + output_spec.interval * (output_spec.num - 1)
    state, outs = fdtd.simulate(
        dt=1.0,
        grid=(jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2))),
        permittivity=jnp.ones((3, 1, 1, 1)),
        conductivity=jnp.zeros((3, 1, 1, 1)),
        source=fdtd.Source(
            offset=(0, 0, 0),
            field=jnp.ones((3, 1, 1, 1)),
            waveform=jnp.ones((num_steps,)),
        ),
        output_spec=output_spec,
    )
    np.testing.assert_array_equal(
        outs[0][:, 0, 0, 0, 0],
        output_spec.start + output_spec.interval * jnp.arange(output_spec.num),
    )
