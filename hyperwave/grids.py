from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .defs import Int3


class Grid(NamedTuple):
    """Defines the grid spacing for the Yee grid."""

    du: Tuple[ArrayLike, ArrayLike, ArrayLike]

    def shape(self) -> Int3:
        return tuple(du.shape[0] for du in self.du)

    def is_valid(self) -> bool:
        return all(du.shape[1] == 2 and du.ndim == 2 for du in self.du)

    def values(self, axis: int, is_forward: bool) -> jax.Array:
        index = 1 if is_forward else 0  # TODO: Not sure if this is correct.
        return jnp.expand_dims(
            self.du[axis][:, index],
            axis=range(axis + 1, 0) if axis < 0 else range(axis + 1, 3),
        )


def spatial_diff(
    field: ArrayLike,
    delta: ArrayLike,
    axis: int,
    is_forward: bool,
) -> jax.Array:
    """Returns the spatial differences of ``field`` along ``axis``."""
    if is_forward:
        return (jnp.roll(field, shift=+1, axis=axis) - field) / delta
    else:
        return (field - jnp.roll(field, shift=-1, axis=axis)) / delta


def curl(field: ArrayLike, grid: Grid, is_forward: bool) -> jax.Array:
    """Returns the curl of ``field`` on ``grid``."""
    fx, fy, fz = [field[..., i, :, :, :] for i in range(3)]
    dx, dy, dz = [
        partial(
            spatial_diff,
            delta=grid.values(axis=a, is_forward=is_forward),
            axis=a,
            is_forward=is_forward,
        )
        for a in range(-3, 0)
    ]
    return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)
