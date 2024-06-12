"""Ensure that ``solve.solve()`` solves the wave equation."""

from __future__ import annotations

import jax.numpy as jnp

from hyperwave import solve


def run_solve(shape, omegas, err_thresh, max_steps):
    xx, yy, zz = shape
    dt = 0.99 / jnp.sqrt(3)
    grid = solve.Grid(dt=dt, du=tuple(jnp.ones((s, 2)) for s in shape))
    epsilon, sigma, source = [jnp.zeros(grid.field_shape())] * 3
    epsilon += 1
    sigma += 6e-2
    source = source.at[2, xx // 2, yy // 2, zz // 2].set(2.0)

    fields, errs, steps, is_success, err_fields, err_hist = solve.solve(
        omegas=omegas,
        epsilon=epsilon,
        sigma=sigma,
        source=source,
        grid=grid,
        err_thresh=err_thresh,
        max_steps=max_steps,
    )

    # for i in range(fields.shape[0]):
    #   plot_complex(fields[i])
    #   plot_complex(err_fields[i])
    #
    #
    # plt.figure()
    # plt.semilogy(err_hist, ".-")
    # plt.ylim(-1e-2, 1e-2)

    print(f"{errs}, {steps}, {is_success}")
    return is_success


def test_run_solve():
    assert run_solve(
        shape=(100, 100, 40),
        omegas=solve.FreqSpace(
            start=2 * jnp.pi / 20,
            stop=2 * jnp.pi / 16,
            num=20,
        ),
        err_thresh=1e-2,
        max_steps=10_000,
    )
