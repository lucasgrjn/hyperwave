from typing import NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import defs, grids, utils


class State(NamedTuple):
    step: int
    e_field: ArrayLike
    h_field: ArrayLike


class OutputRange(NamedTuple):
    start: int
    interval: int
    num: int


Outputs = Tuple[jax.Array]


# @jax.jit
def simulate(
    dt: float,
    grid: grids.Grid,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source_offset: defs.Int3,
    source_field: ArrayLike,
    source_waveform: ArrayLike,
    output_offsets: Sequence[defs.Int3],
    output_shapes: Sequence[defs.Int3],
    output_range: OutputRange,
    state: Optional[State] = None,
) -> Tuple[State, Outputs]:

    # Precomputed update coefficients
    z = conductivity * dt / (2 * permittivity)
    ca = (1 - z) / (1 + z)
    cb = dt / permittivity / (1 + z)

    def step_fn(_, state: State) -> State:
        step, e, h = state

        h = h - dt * grids.curl(e, grid, is_forward=True)

        u = grids.curl(h, grid, is_forward=False)
        u = utils.at(u, source_offset, source_field.shape).add(
            source_field * source_waveform[step]
        )
        e = ca * e + cb * u

        return State(step + 1, e, h)

    def output_fn(index: int, outs: Outputs, e_field: ArrayLike) -> Outputs:
        return tuple(
            out.at[index].set(utils.get(e_field, offset, shape))
            for out, offset, shape in zip(outs, output_offsets, output_shapes)
        )

    def update_and_output(
        state: State, outs: Outputs, output_index: int, num_steps: int
    ) -> Tuple[State, Outputs]:
        state = jax.lax.fori_loop(
            lower=0, upper=num_steps, body_fun=step_fn, init_val=state
        )
        outs = output_fn(output_index, outs, state.e_field)
        return state, outs

    # Initialize state and outputs.
    if state is None:
        state = State(
            step=0,
            e_field=jnp.zeros((3,) + grid.shape()),
            h_field=jnp.zeros((3,) + grid.shape()),
        )
    outs = tuple(jnp.empty((output_range.num, 3) + shape) for shape in output_shapes)

    # Initial update to first output.
    state, outs = update_and_output(
        state, outs, output_index=0, num_steps=output_range.start
    )

    # Materialize the rest of the output snapshots.
    for output_index in range(1, output_range.num):
        state, outs = update_and_output(
            state, outs, output_index, num_steps=output_range.interval
        )

    return state, outs
