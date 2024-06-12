"""Basic definitions."""

from __future__ import annotations

from typing import Tuple

from jax.typing import ArrayLike

# Tuple of 3 integers, used for ``(x, y, z)`` data.
Int3 = Tuple[int, int, int]

# Minimal, sufficient definition of the Yee lattice for the simulation volume.
#
# Defines the distance between adjacent components along x-, y-, and z-axes.
# Each of the three arrays should be of shape ``(uu, 2)`` for each of the 3
# spatial axes, where the ``[:, 1]`` values are shifted by half a cell in the
# positive direction of the axis in relation to the ``[:, 0]`` values.
#
Grid = Tuple[ArrayLike, ArrayLike, ArrayLike]
