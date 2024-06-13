"""Computes the electromagnetic wave equation error."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import grids
from .typing import Grid


def wave_equation_errors(
    fields: ArrayLike,
    omegas: ArrayLike,
    phases: ArrayLike,
    epsilon: ArrayLike,
    sigma: ArrayLike,
    source: ArrayLike,
    grid: Grid,
) -> jax.Array:
    w = jnp.expand_dims(omegas, axis=range(-4, 0))
    phi = jnp.expand_dims(phases, axis=range(-4, 0))
    err = (
        grids.curl(grids.curl(fields, grid, is_forward=True), grid, is_forward=False)
        - w**2 * (epsilon - 1j * sigma / w) * fields
        + 1j * w * source * jnp.exp(1j * phi)
    )
    return (
        jnp.sqrt(jnp.sum(jnp.abs(err) ** 2, axis=(1, 2, 3, 4)))
        / (omegas * jnp.linalg.norm(source))
    ), err


# Compute the error.
