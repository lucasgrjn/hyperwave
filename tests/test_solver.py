"""Ensure that ``solve.solve()`` solves the wave equation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import hyperwave as hw


def test_fdtd_simulation():
    snapshot_range = hw.solver.Range(
        start=10,
        interval=2,
        num=3,
    )
    num_steps = snapshot_range.start + snapshot_range.interval * (
        snapshot_range.num - 1
    )
    state, outs = hw.solver.simulate(
        dt=1.0,
        grid=(jnp.ones((1, 2)), jnp.ones((1, 2)), jnp.ones((1, 2))),
        permittivity=jnp.ones((3, 1, 1, 1)),
        conductivity=jnp.zeros((3, 1, 1, 1)),
        source_field=hw.solver.Subfield(
            offset=(0, 0, 0), field=-jnp.ones((3, 1, 1, 1))
        ),
        source_waveform=jnp.ones((num_steps,)),
        output_volumes=[hw.solver.Volume((0, 0, 0), (2, 1, 1))],
        snapshot_range=snapshot_range,
    )
    assert state.e_field.shape == (3, 1, 1, 1)
    np.testing.assert_array_equal(
        outs[0][:, 0, 0, 0, 0],
        snapshot_range.start
        + snapshot_range.interval * jnp.arange(snapshot_range.num)
        + 1,
    )


# TODO: Test continuity. That is, that we can get the same result with two simulations as with a single simulation.


def run_solve(shape, freq_band, err_thresh, max_steps):
    xx, yy, zz = shape
    grid = tuple(jnp.ones((s, 2)) for s in shape)
    epsilon, sigma, source = [jnp.zeros((3,) + shape)] * 3
    epsilon += 1
    sigma += 6e-2

    source = source.at[2, xx // 2, yy // 2, zz // 2].set(2.0)

    fields, errs, steps, is_success, err_fields, err_hist = hw.solver.solve(
        freq_band=freq_band,
        permittivity=epsilon,
        conductivity=sigma,
        source=hw.solver.Subfield(offset=(0, 0, 0), field=source),
        grid=grid,
        err_thresh=err_thresh,
        max_steps=max_steps,
    )

    print(f"{errs}, {steps}, {is_success}")
    return is_success


def test_run_solve():
    # TODO: Also test that we complete in > 1 iterations?
    assert run_solve(
        shape=(100, 100, 40),
        freq_band=hw.solver.Band(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
        err_thresh=1e-2,
        max_steps=5_000,
    )
