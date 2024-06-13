"""Temporal sampling for efficient frequency component extraction."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .typing import Range


def omegas(freq_range: Range) -> jax.Array:
    start, stop, num = freq_range
    if num == 1:
        return jnp.array([(start + stop) / 2])
    else:
        return jnp.linspace(start, stop, num)


def sampling_interval(freq_range: Range) -> float:
    w = omegas(freq_range)
    if len(w) == 1:
        return float(jnp.pi / (2 * w[0]))
    else:
        w_avg = (w[0] + w[-1]) / 2
        dw = (w[-1] - w[0]) / (len(w) - 1)
        return _round_to_mult(
            2 * jnp.pi / (len(w) * dw),
            multiple=jnp.pi / (len(w) * w_avg),
            offset=0.5,
        )


def _round_to_mult(x, multiple, offset=0):
    return (round(x / multiple - offset) + offset) * multiple


def project(
    snapshots: ArrayLike,
    omegas: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    # Build ``P`` matrix.
    wt = omegas[None, :] * t[:, None]
    P = jnp.concatenate([jnp.cos(wt), -jnp.sin(wt)], axis=1)

    # Project out frequency components.
    res = jnp.einsum("ij,j...->i...", jnp.linalg.inv(P), snapshots)
    return res[: len(omegas)] + 1j * res[len(omegas) :]
