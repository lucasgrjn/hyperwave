"""Definition of and operations on the Yee cell simulation volume.

NOTE: We use a definition of the Yee cell simulation grid (see 
https://en.wikipedia.org/wiki/Finite-difference_time-domain_method) where the
{xyz}-component of the electric field is shifted by a half-cell in the {xyz}
direction as well as along the time axis. The corresponding magnetic field
components are shifted only along the spatial axes perpendicular to the
component axis.

"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import defs

# Minimal, sufficient definition of the Yee lattice for the simulation volume.
#
# Defines the distance between adjacent components along x-, y-, and z-axes.
# Each of the three arrays should be of shape ``(uu, 2)`` for each of the 3
# spatial axes, where the ``[:, 1]`` values are shifted by half a cell in the
# positive direction of the axis in relation to the ``[:, 0]`` values.
#
Grid = Tuple[ArrayLike, ArrayLike, ArrayLike]


def shape(grid: Grid) -> defs.Int3:
    """``(xx, yy, zz)`` shape of the simulation volume according to ``grid``."""
    return tuple(du.shape[0] for du in grid)


def is_valid(grid: Grid) -> bool:
    """``True`` iff ``grid`` is valid."""
    return all(du.shape[1] == 2 and du.ndim == 2 for du in grid)


def spatial_diff(
    field: ArrayLike,
    delta: ArrayLike,
    axis: int,
    is_forward: bool,
) -> jax.Array:
    """Spatial differences of ``field`` along ``axis``."""
    if is_forward:
        return (jnp.roll(field, shift=+1, axis=axis) - field) / delta
    else:
        return (field - jnp.roll(field, shift=-1, axis=axis)) / delta


def curl(field: ArrayLike, grid: Grid, is_forward: bool) -> jax.Array:
    """Curl of ``field`` on ``grid`` with ``is_forward=False`` for E-field."""

    delta_index = 1 if is_forward else 0
    deltas = tuple(
        jnp.expand_dims(grid[a][:, delta_index], range(1, 3 - a)) for a in range(3)
    )

    fx, fy, fz = [field[..., i, :, :, :] for i in range(3)]
    dx, dy, dz = [
        partial(
            spatial_diff,
            delta=deltas[axis],
            axis=axis,
            is_forward=is_forward,
        )
        for axis in range(-3, 0)
    ]
    return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)
