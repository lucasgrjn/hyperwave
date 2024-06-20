"""Solves the wave equation via FDTD simulation."""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import fdtd, sampling, wave_equation
from .typing import Band, Grid, Range, Subfield, Volume


def solve(
    grid: Grid,
    freq_band: Band,
    permittivity: ArrayLike,
    conductivity: ArrayLike,
    source: Subfield,
    err_thresh: float | None,
    max_steps: int,
    output_volumes: Sequence[Volume] | None = None,
) -> Tuple[jax.Array | Tuple[jax.Array, ...], jax.Array | None, int]:
    r"""Solve the time-harmonic electromagnetic wave equation.

    :py:func:`solve` attempts to solve
    :math:`\nabla \times \nabla \times E - \omega^2 \epsilon E = i \omega J`
    where

    * :math:`\nabla \times` is the `curl operation <https://en.wikipedia.org/wiki/Curl_(mathematics)>`_,
    * :math:`E` is the electric-field,
    * :math:`\omega` is the angular frequency,
    * :math:`\epsilon` is the relative permittivity, and
    * :math:`J` is the electric current excitation.

    Dimensionsless units are used such that

    * :math:`\epsilon_0 = 1` and :math:`\mu_0 = 1`, the permittivity and
      permeability of vacuum are set to (dimensionless) :math:`1`,
    * :math:`c = 1`, the speed of light is also equal (dimensionless) :math:`1`,
    * space is also dimensionless, and
    * :math:`\omega = 2 \pi / \lambda_0`, the angular frequency can be defined
      relative to (dimensionless) vacuum wavelength.

    :math:`E`, :math:`\epsilon`, and :math:`J` values are located on the
    `Yee lattice <https://en.wikipedia.org/wiki/Finite-difference_time-domain_method>`_,
    at locations corresponding to the electric field position.

    Supports simple dielectric materials (no disperson or nonlinearity, loss
    supported) via :math:`\epsilon = \epsilon' + i \sigma / \omega`, where
    :math:`\epsilon'` is the real-valued permittivity and :math:`\sigma` is the
    real-valued conductivity. Anisotropy is explicitly supported by virtue that
    the components of :math:`\epsilon` can be arbitrarily valued, although
    off-diagonal values of the permittivity tensor are not supported.

    :py:func:`solve` can obtain solutions to the wave equation for multiple
    regularly-spaced angular frequencies simultaneously; however, each frequency
    is not allowed to have its own independent input current source.

    The error in the wave equation is defined as
    :math:`(1 / \sqrt{n}) \cdot \lVert \nabla \times \nabla \times E - \omega^2 \epsilon E - i \omega J \rVert / \lVert \omega J \rVert`
    where :math:`n` is the number of elements in the solution field :math:`E`.

    For very large simulations (possibly over many frequencies), we may not
    want to store the entirety of the solution fields. In these cases, the
    error computation can be elided and we can choose to return only the desired
    subdomains of the output fields.

    Args:
        grid: Spacing of the simulation grid.
        freq_band: Angular frequencies at which to produce solution fields.
        permittivity: ``(3, xx, yy, zz)`` array of real-valued permittivity
          values.
        conductivity: ``(3, xx, yy, zz)`` array of real-valued conductivity
          values.
        source: Complex-valued current excitation.
        err_thresh: Terminate when the error of the fields at each frequency
          are below this value. If ``None``, do not compute error values and
          instead terminate on the ``max_steps`` condition only. For
          ``err_thresh <= 0``, termination on ``max_steps`` is guaranteed
          without omitting the error computation.
        max_steps: Maximum number of simulation update steps to execute.
          This is a "soft" termination barrier as additional steps beyond
          ``max_steps`` may be needed to extract solution fields.
        output_volumes: If ``None`` (default), then the solution fields are
          returned in their entirety; otherwise, only sub-volumes of the
          solution field are returned.

    Returns:
        ``(outs, errs, num_steps)`` where

        * when ``output_volumes=None``, ``outs`` is a ``(ww, 3, xx, yy, zz)``
          array (where ``ww`` is the number of frequencies requested via
          ``freq_band.num``).
        * when ``output_volumes`` is a n-tuple of :py:class:`Volume` objects,
          ``outs`` is an n-tuple of ``(ww, 3, xxi, yyi, zzi)`` corresponding to
          the ``shape`` parameters in ``output_volumes``.
        * ``errs`` is a ``(ww,)`` array at each frequency, or else ``None`` for
          the case where ``err_thresh=None``.
        * ``num_steps`` is the number of time-domain updates executed.

    """
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
