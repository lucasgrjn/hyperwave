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

from . import defs, grids, utils


class State(NamedTuple):
    """State of the simulation with E-field a half-step ahead of H-field."""

    step: int
    e_field: ArrayLike
    h_field: ArrayLike


class Source(NamedTuple):
    """Current source to inject into the simulation."""

    offset: defs.Int3  # Location to inject source in simulation volume.
    field: ArrayLike  # ``(3, xx0, yy0, zz0)`` complex-valued source field.
    waveform: ArrayLike  # ``(tt,)`` complex-valued source amplitude.

    def inject(self, e_field: ArrayLike, step: int) -> jax.Array:
        return utils.at(e_field, self.offset, self.field.shape).add(
            self.field * self.waveform[step]
        )


class OutputSpec(NamedTuple):
    """E-field subvolumes to snapshot from simulation."""

    start: int
    interval: int
    num: int
    offsets: Sequence[defs.Int3]
    shapes: Sequence[defs.Int3]


# Convenience type alias for simulation outputs.
Outputs = Tuple[jax.Array, ...]


def simulate(
    dt: float,
    grid: grids.Grid,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: Source,
    output_spec: OutputSpec,
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
        output_spec: Defines E-field snapshots to return, relative to the time
          step of the initial state of the simulation.
        state: Initial state of the simulation.

    Returns:
      ``(state, outputs)`` corresponding to updated simulation state and output
      fields corresponding to ``output_spec``.
    """

    # Precomputed update coefficients
    z = conductivity * dt / (2 * permittivity)
    ca = (1 - z) / (1 + z)
    cb = dt / permittivity / (1 + z)

    def step_fn(_, state: State) -> State:
        """``state`` evolved by one FDTD update."""
        step, e, h = state
        h = h - dt * grids.curl(e, grid, is_forward=True)
        e = ca * e + cb * source.inject(grids.curl(h, grid, is_forward=False), step)
        return State(step + 1, e, h)

    def output_fn(index: int, outs: Outputs, e_field: ArrayLike) -> Outputs:
        """``outs`` updated at ``index`` with ``e_field``."""
        return tuple(
            out.at[index].set(utils.get(e_field, offset, shape))
            for out, offset, shape in zip(outs, output_spec.offsets, output_spec.shapes)
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
            step=0,
            e_field=jnp.zeros((3,) + grids.shape(grid)),
            h_field=jnp.zeros((3,) + grids.shape(grid)),
        )
    outs = tuple(
        jnp.empty((output_spec.num, 3) + shape) for shape in output_spec.shapes
    )

    # Initial update to first output.
    state, outs = update_and_output(
        state, outs, output_index=0, num_steps=output_spec.start
    )

    # Materialize the rest of the output snapshots.
    for output_index in range(1, output_spec.num):
        state, outs = update_and_output(
            state, outs, output_index, num_steps=output_spec.interval
        )

    return state, outs
