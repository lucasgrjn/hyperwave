"""Solves the wave equation via FDTD simulation."""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import fdtd, sampling, wave_equation
from .typing import Grid, Range

# Type alias for jax function inputs. See
# https://jax.readthedocs.io/en/latest/jax.typing.html#jax-typing-best-practices
# for additional information.
ArrayLike = jax.typing.ArrayLike

# Type aliases.
Index = Tuple[int, int, int]
Shape = Tuple[int, int, int]
Offset = Tuple[int, int, int]
FieldShape = Tuple[int, int, int, int]


class SnapshotRange(NamedTuple):
    start: int
    interval: int
    num: int


class OutputSpec(NamedTuple):
    offset: Offset
    shape: Shape
    range: SnapshotRange


class InputSpec(NamedTuple):
    offset: Offset
    field: ArrayLike
    waveform: ArrayLike


def source_waveform(
    t: ArrayLike,
    omegas: ArrayLike,
    phases: ArrayLike,
    rise_time: float,
) -> jax.Array:
    # NOTE: ``rise_time`` refers to the amount of time taken for the sigmoid
    # envelope function to go from an amplitude of roughly ``0.01`` to ``0.99``.

    # NOTE: This happens when we go from ``jax.nn.sigmoid(-4.6)`` to
    # ``jax.nn.sigmoid(+4.6)``.

    return (jax.nn.sigmoid(4.6 * (2 * t / rise_time - 1))) * jnp.sum(
        (jnp.exp(1j * (omegas[:, None] * t + phases[:, None]))), axis=0
    )


def simulate_newer(
    epsilon: ArrayLike,
    sigma: ArrayLike,
    sources: Sequence[InputSpec],
    outputs: Sequence[OutputSpec],
    grid: Grid,
    dt: ArrayLike,
    state: fdtd.State | None = None,
) -> Tuple[jax.Array, jax.Array, Tuple[jax.Array, ...]]:

    state, outs = fdtd.simulate(
        dt=dt,
        grid=grid,
        permittivity=epsilon,
        conductivity=sigma,
        source=fdtd.Source(
            offset=sources[0].offset,
            field=sources[0].field,
            waveform=sources[0].waveform,
        ),
        output_spec=fdtd.OutputSpec(
            offsets=[outputs[0].offset],
            shapes=[outputs[0].shape],
            start=outputs[0].range.start,
            interval=outputs[0].range.interval,
            num=outputs[0].range.num,
        ),
        state=state,
    )
    return state, outs


# TODO: Need to figure out the API here. Something like
# output=None --> return full outputs, otherwise return specific subvolumes.
# err_thresh=None --> Don't compute error, just go directly for ``max_steps``.


# TODO: Need to include phases for the source somewhere here.
def solve(
    # omegas: sampling.FreqSpace,
    freq_range: Range,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: ArrayLike,  # TODO: More general structure that has a time-freq connection.
    # TODO: Add output subvolumes
    grid: Grid,
    err_thresh: float,
    max_steps: int,
) -> Tuple[jax.Array, jax.Array, int, bool]:
    """Solution fields and error for the wave equation at ``omegas``."""
    shape = permittivity.shape[-3:]  # TODO: Do better.

    omegas = sampling.omegas(freq_range)
    sampling_interval = sampling.sampling_interval(freq_range)

    # Steps to sample against.
    dt = 0.99 * jnp.min(permittivity) / jnp.sqrt(3)
    if len(omegas) > 1:  # Adjust dt.
        n = int(jnp.floor(sampling_interval / dt))
        dt = sampling_interval / (n + 1)
        sample_every_n_steps = n + 1
    else:
        sample_every_n_steps = int(round(sampling_interval / dt))
    sample_steps = sample_every_n_steps * (2 * len(omegas) - 1) + 1

    # Heuristic for determining the number of steps to simulate for, before
    # checking for the termination condition.

    # TODO: Think about how long to run the simulation.
    min_sim_steps_heuristic = sum(shape)  # TODO: Move constant.

    # Actual number of steps to simulate.
    steps_per_sim = max(min_sim_steps_heuristic, sample_steps)
    # steps_per_sim = sample_steps

    # Reset ``max_steps`` for what the actual maximum number of steps will be.
    if max_steps % steps_per_sim != 0:
        max_steps = max_steps + steps_per_sim - (max_steps % steps_per_sim)

    # TODO: Do something for phases here.
    phases = -1 * jnp.pi * jnp.arange(len(omegas))

    # Form input for simulation.
    inputspec = InputSpec(
        offset=(0, 0, 0),
        field=source,
        waveform=source_waveform(
            t=dt * jnp.arange(max_steps),
            omegas=omegas,
            phases=phases,
            rise_time=1,  # TODO: Put constant somewhere better.
        ),
    )

    # Initial values.
    e_field, h_field = 2 * [jnp.zeros_like(permittivity)]

    errs_hist = []  # TODO: Remove.

    state = None

    for start_step in range(0, max_steps, steps_per_sim):
        outputspec = OutputSpec(
            offset=(0, 0, 0),
            shape=shape,
            range=SnapshotRange(
                start=start_step + steps_per_sim - sample_steps,
                interval=sample_every_n_steps,
                num=2 * len(omegas),
            ),
        )

        # Run simulation.
        state, outs = simulate_newer(
            state=state,
            epsilon=permittivity,
            sigma=conductivity,
            sources=[inputspec],
            outputs=[outputspec],
            grid=grid,
            dt=dt,
        )

        # Infer time-harmonic fields.
        # TODO: Generalize beyond 1st output.
        t = dt * (
            0.5
            + outputspec.range.start
            + outputspec.range.interval * jnp.arange(outputspec.range.num)
        )
        freq_fields = sampling.project(outs[0], omegas, t)

        # Compute error.
        errs, err_fields = wave_equation.wave_equation_errors(
            fields=freq_fields,
            omegas=omegas,
            phases=phases,
            epsilon=permittivity,
            sigma=conductivity,
            source=source,
            grid=grid,
        )
        errs_hist.append(errs)

        if jnp.max(errs) < err_thresh:
            return (
                freq_fields,
                errs,
                start_step + steps_per_sim,
                True,
                err_fields,
                errs_hist,
            )

    return freq_fields, errs, max_steps, False, err_fields, errs_hist
