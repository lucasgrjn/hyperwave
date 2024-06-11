import jax.numpy as jnp
import numpy as np
import pytest

from hyperwave import fdtd


# TODO: Remove? Not sure if we make this part of the API?
@pytest.mark.parametrize("output_range", [fdtd.OutputRange(10, 2, 3)])
def test_fdtd_simulation(output_range: fdtd.OutputRange):
    num_steps = output_range.start + output_range.interval * (output_range.num - 1)
    state, outs = fdtd.simulate(
        dt=1.0,
        grid=(jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2))),
        permittivity=jnp.ones((3, 1, 1, 1)),
        conductivity=jnp.zeros((3, 1, 1, 1)),
        source_offset=(0, 0, 0),
        source_field=jnp.ones((3, 1, 1, 1)),
        source_waveform=jnp.ones((num_steps,)),
        output_offsets=[(0, 0, 0)],
        output_shapes=[(1, 1, 1)],
        output_range=output_range,
    )
    np.testing.assert_array_equal(
        outs[0][:, 0, 0, 0, 0],
        output_range.start + output_range.interval * jnp.arange(output_range.num),
    )
