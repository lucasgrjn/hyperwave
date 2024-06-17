"""Basic definitions."""

from __future__ import annotations

from typing import NamedTuple, Tuple

from jax.typing import ArrayLike

# NOTE: Please avoid including logic here! Included types should be trivially simple.

# Tuple of 3 integers, used for ``(x, y, z)`` data.
Int3 = Tuple[int, int, int]


class Subfield(NamedTuple):
    """Field defined at ``offset`` in space."""

    offset: Int3
    field: ArrayLike


class Volume(NamedTuple):
    """Identifies a volume of size ``shape`` at ``offset`` in space."""

    offset: Int3
    shape: Int3


class Range(NamedTuple):
    """Describes values ``start + i * interval`` for ``i`` in ``[0, num)``."""

    start: int
    interval: int
    num: int


class Band(NamedTuple):
    """Describes ``num`` regularly spaced values within ``[start, stop].``"""

    start: float
    stop: float
    num: int


# Minimal, sufficient definition of the Yee lattice for the simulation volume.
#
# Defines the distance between adjacent components along x-, y-, and z-axes.
# Each of the three arrays should be of shape ``(uu, 2)`` for each of the 3
# spatial axes, where the ``[:, 1]`` values are shifted by half a cell in the
# positive direction of the axis in relation to the ``[:, 0]`` values.
#
Grid = Tuple[ArrayLike, ArrayLike, ArrayLike]
