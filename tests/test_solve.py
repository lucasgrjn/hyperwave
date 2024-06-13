"""Ensure that ``solve.solve()`` solves the wave equation."""

from __future__ import annotations

import jax.numpy as jnp

from hyperwave import solve


def run_solve(shape, freq_range, err_thresh, max_steps):
    xx, yy, zz = shape
    grid = tuple(jnp.ones((s, 2)) for s in shape)
    epsilon, sigma, source = [jnp.zeros((3,) + shape)] * 3
    epsilon += 1
    sigma += 6e-2
    source = source.at[2, xx // 2, yy // 2, zz // 2].set(2.0)

    fields, errs, steps, is_success, err_fields, err_hist = solve.solve(
        freq_range=freq_range,
        permittivity=epsilon,
        conductivity=sigma,
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
    # TODO: Also test that we complete in > 1 iterations?
    assert run_solve(
        shape=(100, 100, 40),
        freq_range=(2 * jnp.pi / 20.0, 2 * jnp.pi / 16.0, 20),
        err_thresh=1e-2,
        max_steps=5_000,
    )


# def test_comp_sim():
#     dt = 1.0
#     shape = (1, 1, 1)
#     grid = solve.Grid(dt=dt, du=tuple(jnp.ones((s, 2)) for s in shape))
#     e_field = jnp.zeros((3,) + shape)
#     h_field = jnp.zeros((3,) + shape)
#     epsilon = jnp.ones((3,) + shape)
#     sigma = jnp.zeros((3,) + shape)
#     source = jnp.zeros((3,) + shape)
#     source = source.at[2, 0, 0, 0].set(1.0)
#     start_step = 0
#     outputspec = solve.OutputSpec(
#         offset=(0, 0, 0),
#         shape=grid.shape(),
#         range=solve.SnapshotRange(
#             start=0,
#             interval=4,
#             num=3,
#         ),
#     )
#     num_steps = (
#         outputspec.range.start
#         + (outputspec.range.num - 1) * outputspec.range.interval
#         + 1
#     )
#     inputspec = solve.InputSpec(
#         offset=(0, 0, 0),
#         field=source,
#         waveform=-jnp.ones((num_steps + 1,)),
#         # waveform=solve.source_waveform(
#         #     t=grid.dt * jnp.arange(max_steps),
#         #     omegas=omegas,
#         #     rise_time=1,  # TODO: Put constant somewhere better.
#         # ),
#     )
#     e_field_0, h_field_0, outs_0 = solve.simulate(
#         e_field=e_field,
#         h_field=h_field,
#         epsilon=epsilon,
#         sigma=sigma,
#         sources=[inputspec],
#         outputs=[outputspec],
#         grid=grid,
#         start_step=start_step,
#         num_steps=num_steps,
#     )
#     e_field_1, h_field_1, outs_1 = solve.simulate_new(
#         e_field=e_field,
#         h_field=h_field,
#         epsilon=epsilon,
#         sigma=sigma,
#         sources=[inputspec],
#         outputs=[outputspec],
#         grid=grid,
#         start_step=start_step,
#         num_steps=num_steps,
#     )
#     np.testing.assert_array_equal(e_field_0, e_field_1)
#     np.testing.assert_array_equal(h_field_0, h_field_1)
#     np.testing.assert_array_equal(outs_0[0][:, 2, 0, 0, 0], outs_1[0][:, 2, 0, 0, 0])
