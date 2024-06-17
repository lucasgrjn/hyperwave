"""Implementation of FDTD method.

A simple, feature-minimal implementation of the finite-difference time-domain
method. Serves as the underlying simulation method for solving the wave
equation.

Strategically used as an internal API in anticipation of allowing alternate,
optimized FDTD routines to be used in the place of the one outlined here.

"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import grids, utils
from .typing import Grid, Range, Subfield, Volume


class State(NamedTuple):
    """State of the simulation with E-field a half-step ahead of H-field."""

    step: int
    e_field: ArrayLike
    h_field: ArrayLike


# Convenience type alias for simulation outputs.
Outputs = Tuple[jax.Array, ...]


# TODO: Add TF/SF source case.
def simulate(
    dt: ArrayLike,
    grid: Grid,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    # source: Source,
    source_field: Subfield,
    source_waveform: ArrayLike,
    output_volumes: Sequence[Volume],
    snapshot_range: Range,
    state: State | None = None,
) -> Tuple[State, Outputs]:
    """Execute the finite-difference time-domain (FDTD) simulation method.

    Utilizes dimensionless units where the permittivity and susceptibility of
    vacuum are set to ``1`` as well as dimensionless units for space and time.

    Args:
        dt: Dimensionless value of the amount of time advanced per FDTD update.
        grid: 3-tuple of arrays defining the simulation grid along the spatial
          axes. Each array must be of shape ``(:, 2)`` where the ``[:, 1]``
          values correspond to component spacings which are shifted by a
          half-cell along the positive axis direction relative to the ``[:, 0]``
          values.
        permittivity: ``(3, xx, yy, zz)`` array of (relative) permittivity
          values.
        conductivity: ``(3, xx, yy, zz)`` array of conductivity values.
        source: Current source to inject in the simulation.
        output_volumes: E-field subvolumes of the simulation space to return.
        snapshot_range: Interval of regularly-spaced time steps at which to
          generate output volumes.
        state: Initial state of the simulation. Defaults to field values of
          ``0`` everywhere at ``step = -1``.

    Returns:
      ``(state, outputs)`` corresponding to updated simulation state and output
      fields corresponding to ``output_spec``.
    """

    # TODO: Do some input verification here?

    # Precomputed update coefficients
    z = conductivity * dt / (2 * permittivity)
    ca = (1 - z) / (1 + z)
    cb = dt / permittivity / (1 + z)

    def inject_source(field: ArrayLike, step: ArrayLike) -> ArrayLike:
        return utils.at(field, source_field.offset, source_field.field.shape[-3:]).add(
            -jnp.real(source_field.field * source_waveform[step])
        )

    def step_fn(_, state: State) -> State:
        """``state`` evolved by one FDTD update."""
        step, e, h = state
        h = h - dt * grids.curl(e, grid, is_forward=True)

        # def inject(self, e_field: ArrayLike, step: int) -> jax.Array:
        #     return utils.at(e_field, self.offset, self.field.shape[-3:]).add(
        #         -jnp.real(self.field * self.waveform[step])
        #     )
        e = ca * e + cb * inject_source(grids.curl(h, grid, is_forward=False), step + 1)
        return State(step + 1, e, h)

    def output_fn(index: int, outs: Outputs, e_field: ArrayLike) -> Outputs:
        """``outs`` updated at ``index`` with ``e_field``."""
        return tuple(
            out.at[index].set(utils.get(e_field, ov.offset, ov.shape))
            for out, ov in zip(outs, output_volumes)
        )

    def update_and_output(
        state: State, outs: Outputs, output_index: int, num_steps: int
    ) -> Tuple[State, Outputs]:
        """``num_steps`` updates on ``state`` with ``output_index`` snapshot."""
        state = jax.lax.fori_loop(
            lower=0, upper=num_steps, body_fun=step_fn, init_val=state
        )
        outs = output_fn(output_index, outs, state.e_field)
        return state, outs

    # Initialize initial state and outputs.
    if state is None:
        state = State(
            step=-1,
            e_field=jnp.zeros((3,) + grids.shape(grid)),
            h_field=jnp.zeros((3,) + grids.shape(grid)),
        )

    outs = tuple(jnp.empty((snapshot_range.num, 3) + ov.shape) for ov in output_volumes)

    # Initial update to first output.
    state, outs = update_and_output(
        state, outs, output_index=0, num_steps=snapshot_range.start - state.step
    )

    # Materialize the rest of the output snapshots.
    for output_index in range(1, snapshot_range.num):
        state, outs = update_and_output(
            state, outs, output_index, num_steps=snapshot_range.interval
        )

    return state, outs
