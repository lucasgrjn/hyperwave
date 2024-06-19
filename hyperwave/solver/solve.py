"""Solves the wave equation via FDTD simulation."""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import fdtd, sampling, wave_equation
from .typing import Band, Grid, Range, Subfield, Volume

# TODO: Need to figure out the API here. Something like
# output=None --> return full outputs, otherwise return specific subvolumes.
# err_thresh=None --> Don't compute error, just go directly for ``max_steps``.


# TODO: Need to include phases for the source somewhere here.
def solve(
    grid: Grid,
    freq_band: Band,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: Subfield,  # Consider `` | ArrayLike``
    err_thresh: float,
    max_steps: int,
    output_volumes: Sequence[Volume] | None = None,
) -> Tuple[jax.Array, jax.Array, int, bool]:
    """Solution fields and error for the wave equation at ``omegas``."""
    shape = permittivity.shape[-3:]  # TODO: Do better.

    omegas = sampling.omegas(freq_band)
    sampling_interval = sampling.sampling_interval(freq_band)

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

    # # TODO: Do something for phases here.
    # phases = -1 * jnp.pi * jnp.arange(len(omegas))

    # src = source.as_fdtd(omegas, dt * jnp.arange(max_steps))  # TODO: +1 for max_steps?
    phases = -1 * jnp.pi * jnp.arange(len(omegas))
    t = jnp.arange(max_steps) * dt
    phi = omegas[:, None] * t + phases[:, None]
    waveform = jnp.sum(jnp.exp(1j * phi), axis=0)

    # Initial values.
    e_field, h_field = 2 * [jnp.zeros_like(permittivity)]

    errs_hist = []  # TODO: Remove.

    state = None
    output_volumes = [Volume(offset=(0, 0, 0), shape=shape)]
    for start_step in range(0, max_steps, steps_per_sim):
        snapshot_range = Range(
            start=start_step + steps_per_sim - sample_steps,
            interval=sample_every_n_steps,
            num=2 * len(omegas),
        )

        # Run simulation.
        state, outs = fdtd.simulate(
            dt=dt,
            grid=grid,
            permittivity=permittivity,
            conductivity=conductivity,
            # source=src,
            source_field=source,
            source_waveform=waveform,
            output_volumes=output_volumes,
            snapshot_range=snapshot_range,
            state=state,
        )

        # Infer time-harmonic fields.
        # TODO: Generalize beyond 1st output.
        t = dt * (
            0.5
            + snapshot_range.start
            + snapshot_range.interval * jnp.arange(snapshot_range.num)
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
