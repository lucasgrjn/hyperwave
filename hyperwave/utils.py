"""Utility functions."""

from __future__ import annotations

from jax.typing import ArrayLike

from .typing import Int3


def at(field: ArrayLike, offset: Int3, shape: Int3):
    """Modify ``shape`` values of ``field`` at ``offset``."""
    return field.at[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]


def get(field: ArrayLike, offset: Int3, shape: Int3):
    """Returns ``shape`` values of ``field`` at ``offset``."""
    return field[
        ...,
        offset[0] : offset[0] + shape[0],
        offset[1] : offset[1] + shape[1],
        offset[2] : offset[2] + shape[2],
    ]
