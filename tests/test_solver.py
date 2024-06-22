"""Ensure that ``solve.solve()`` solves the wave equation."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import hyperwave as hw

# TODO: Test basic validation.

# TODO: Get to faster unit testing of ``solve()``.

# TODO: Implement testing plan
# 1. Go to very fast unit tests (these will just test convergence)
# 2. Validation testing (just try to help the user by catching badly shaped inputs)
# 3. Functionality testing of err_thresh and output_volumes things
# 4. Simple integration test that compares wave_equation_error with the error from solve()
# 5. Unit tests to help freeze the FDTD API


@pytest.mark.parametrize(
    "shape,err_thresh,max_steps",
    [
        ((100, 10, 10), 1e-2, 10_000),
    ],
)
def test_solve_convergence(
    shape,
    err_thresh,
    max_steps,
    freq_band=hw.solver.Band(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
):
    xx, yy, zz = shape
    grid = hw.solver.Grid(*[jnp.ones((s, 2)) for s in shape])
    permittivity, conductivity, source = [jnp.zeros((3,) + shape)] * 3
    permittivity += 1
    conductivity += 6e-2

    source = source.at[2, xx // 2, yy // 2, zz // 2].set(2.0)

    fields, errs, steps = hw.solver.solve(
        grid=grid,
        freq_band=freq_band,
        permittivity=permittivity,
        conductivity=conductivity,
        source=hw.solver.Subfield(offset=(0, 0, 0), field=source),
        err_thresh=err_thresh,
        max_steps=max_steps,
    )
    assert all(errs < err_thresh) and steps < max_steps


def test_solve_invalid_inputs():
    with pytest.raises(ValueError, match=r"grid spacings"):
        hw.solver.solve(
            grid=hw.solver.Grid(
                dx=jnp.ones((10, 1)),  # Error here.
                dy=jnp.ones((20, 2)),
                dz=jnp.ones((30, 2)),
            ),
            freq_band=hw.solver.Band(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
            permittivity=jnp.ones((3, 10, 20, 30)),
            conductivity=jnp.zeros((3, 10, 20, 30)),
            source=hw.solver.Subfield(
                offset=(0, 0, 0), field=jnp.zeros((3, 10, 20, 30))
            ),
            err_thresh=1e-2,
            max_steps=10_000,
        )

    with pytest.raises(ValueError, match=r"Permittivity"):
        hw.solver.solve(
            grid=hw.solver.Grid(
                dx=jnp.ones((10, 2)),
                dy=jnp.ones((20, 2)),
                dz=jnp.ones((30, 2)),
            ),
            freq_band=hw.solver.Band(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
            permittivity=jnp.ones((3, 11, 20, 30)),  # Error here.
            conductivity=jnp.zeros((3, 10, 20, 30)),
            source=hw.solver.Subfield(
                offset=(0, 0, 0), field=jnp.zeros((3, 10, 20, 30))
            ),
            err_thresh=1e-2,
            max_steps=10_000,
        )

    with pytest.raises(ValueError, match=r"Source"):
        hw.solver.solve(
            grid=hw.solver.Grid(
                dx=jnp.ones((10, 2)),
                dy=jnp.ones((20, 2)),
                dz=jnp.ones((30, 2)),
            ),
            freq_band=hw.solver.Band(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
            permittivity=jnp.ones((3, 10, 20, 30)),
            conductivity=jnp.zeros((3, 10, 20, 30)),
            source=hw.solver.Subfield(
                offset=(5, 0, 0), field=jnp.zeros((3, 6, 20, 30))  # Error here.
            ),
            err_thresh=1e-2,
            max_steps=10_000,
        )


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
        grid=hw.solver.Grid(
            dx=jnp.ones((1, 2)), dy=jnp.ones((1, 2)), dz=jnp.ones((1, 2))
        ),
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
