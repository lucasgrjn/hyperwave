from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

Int3 = Tuple[int, int, int]  # Used for ``(x, y, z)`` tuple.


def at(field: ArrayLike, offset: Int3, shape: Int3):
    return field.at[
        :,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def get(field: ArrayLike, offset: Int3, shape: Int3):
    return field[
        :,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


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
