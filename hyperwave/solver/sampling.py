"""Temporal sampling for efficient frequency component extraction."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .typing import Band


def project(
    snapshots: ArrayLike,
    omegas: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    """Project ``snapshots`` at ``t`` to angular frequencies ``omegas``."""
    # Build ``P`` matrix.
    wt = omegas[None, :] * t[:, None]
    P = jnp.concatenate([jnp.cos(wt), -jnp.sin(wt)], axis=1)

    # Project out frequency components.
    res = jnp.einsum("ij,j...->i...", jnp.linalg.inv(P), snapshots)
    return res[: len(omegas)] + 1j * res[len(omegas) :]


def sampling_interval(freq_band: Band) -> float:
    """Snapshot interval for efficiently sampling ``freq_band`` frequencies."""
    w = band_values(freq_band)
    if len(w) == 1:
        # For a single frequency, we simply use the quarter-period interval.
        return float(jnp.pi / (2 * w[0]))
    else:
        # For multiple frequencies we sample at the value of ``i`` in
        # ``π * (i + 0.5) / (n * w_avg)`` nearest to the ``2π / (n * dw)``
        # point, where ``w_avg`` is the average angular frequency and ``dw`` is
        # the spacing between neighboring frequencies.
        w_avg = (w[0] + w[-1]) / 2
        dw = (w[-1] - w[0]) / (len(w) - 1)
        return _round_to_mult(
            2 * jnp.pi / (len(w) * dw),
            multiple=jnp.pi / (len(w) * w_avg),
            offset=0.5,
        )


def band_values(band: Band) -> jax.Array:
    """Values represented by ``band``."""
    if band.num == 1:
        return jnp.array([(band.start + band.stop) / 2])
    else:
        return jnp.linspace(band.start, band.stop, band.num)


def _round_to_mult(x, multiple, offset=0):
    return (round(x / multiple - offset) + offset) * multiple
