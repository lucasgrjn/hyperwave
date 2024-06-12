"""Solves the wave equation via FDTD simulation."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp

from . import fdtd

# Type alias for jax function inputs. See
# https://jax.readthedocs.io/en/latest/jax.typing.html#jax-typing-best-practices
# for additional information.
ArrayLike = jax.typing.ArrayLike

# Type aliases.
Index = Tuple[int, int, int]
Shape = Tuple[int, int, int]
Offset = Tuple[int, int, int]
FieldShape = Tuple[int, int, int, int]


class Grid(NamedTuple):
    """Defines the FDTD simulation grid."""

    # Time difference between FDTD updates.
    dt: float

    # ``dx, dy, dz = du`` defines the spatial grid, where each ``du`` array is of
    # shape ``(uu, 2)`` where the ``du[:, 0]`` and ``du[:, 1]`` indices represent
    # the spatial differences centered at the ``u = u_0`` and ``u = u_0 + 0.5``
    # Yee grid locations. That is, ``du[i, 1]`` is located between ``du[i, 0]``
    # and ``du[i + 1, 0]`` so that we define the grid spacingsat every half-grid
    # locations (this is necessary because of the "half-offset" nature of the Yee
    # cell).
    du: Tuple[ArrayLike, ArrayLike, ArrayLike]

    def shape(self) -> Shape:
        """``(xx, yy, zz)`` of simulation domain."""
        return tuple(du.shape[0] for du in self.du)

    def field_shape(self) -> FieldShape:
        """``(3, xx, yy, zz)`` of simulation domain."""
        return (3,) + self.shape()

    def expanded_delta(self, axis: int, is_forward: bool) -> jax.Array:
        """ "Expanded array for the grid spacing along ``axis <= 0``."""
        assert axis <= 0
        ind = 0 if is_forward else 1
        return jnp.expand_dims(self.du[axis][:, ind], axis=range(axis + 1, 0))


class FreqSpace(NamedTuple):
    start: float
    stop: float
    num: int

    @property
    def freqs(self) -> jax.Array:
        if self.num == 1:
            return jnp.array([(self.start + self.stop) / 2])
        else:
            return jnp.linspace(self.start, self.stop, self.num)

    @property
    def phases(self) -> jax.Array:
        # return jnp.zeros((self.num,))
        return -1 * jnp.linspace(
            start=0,
            stop=self.num * jnp.pi,
            endpoint=False,
            num=self.num,
        )

    @property
    def sampling_interval(self) -> float:
        w_avg = (self.start + self.stop) / 2
        if self.num == 1:
            return float(jnp.pi / (2 * w_avg))  # Quarter-period.
        else:
            dw = (self.stop - self.start) / (self.num - 1)
            return float(
                self.round_to_mult(
                    2 * jnp.pi / (self.num * dw),
                    multiple=jnp.pi / (self.num * w_avg),
                    offset=0.5,
                )
            )

    @staticmethod
    def round_to_mult(x, multiple, offset=0):
        # return x
        return (round(x / multiple - offset) + offset) * multiple


class SnapshotRange(NamedTuple):
    start: int
    interval: int
    num: int

    def arange(self) -> jax.Array:
        return jnp.arange(
            start=self.start,
            stop=self.start + self.interval * self.num,
            step=self.interval,
        )


def subvolume_inds(offset: Offset, shape: Shape) -> Tuple[Index, Index]:
    return offset, tuple(a + b for a, b in zip(offset, shape))


class OutputSpec(NamedTuple):
    offset: Offset
    shape: Shape
    range: SnapshotRange

    def init_array(self) -> jax.Array:
        return jnp.empty((self.range.num, 3) + self.shape)

    def update(self, arr: ArrayLike, field: ArrayLike, step: int) -> jax.Array:
        start, interval, num = self.range
        is_in_range = jnp.logical_and(
            jnp.greater_equal(step, start),
            jnp.less_equal(step, start + (num - 1) * interval),
        )
        is_modulo = jnp.equal((step - start) % interval, 0)
        i, j = subvolume_inds(self.offset, self.shape)
        return jax.lax.select(
            pred=jnp.logical_and(is_in_range, is_modulo),
            on_false=arr,
            on_true=arr.at[(step - start) // interval].set(
                field[:, i[0] : j[0], i[1] : j[1], i[2] : j[2]]
            ),
        )


class InputSpec(NamedTuple):
    offset: Offset
    field: ArrayLike
    waveform: ArrayLike

    def inject(self, field: ArrayLike, step: int) -> jax.Array:
        i, j = subvolume_inds(self.offset, self.field.shape[-3:])
        return field.at[:, i[0] : j[0], i[1] : j[1], i[2] : j[2]].add(
            -jnp.real(self.waveform[step] * self.field)
        )


def spatial_diff(
    field: ArrayLike,
    delta: ArrayLike,
    axis: int,
    is_forward: bool,
) -> jax.Array:
    """Returns the spatial differences of ``field`` along ``axis``."""
    if is_forward:
        return (jnp.roll(field, shift=+1, axis=axis) - field) / delta
    else:
        return (field - jnp.roll(field, shift=-1, axis=axis)) / delta


def curl(field: ArrayLike, grid: Grid, is_forward: bool) -> jax.Array:
    """Returns the curl of ``field`` on ``grid``."""
    fx, fy, fz = [field[..., i, :, :, :] for i in range(3)]
    dx, dy, dz = [
        partial(
            spatial_diff,
            delta=grid.expanded_delta(axis=a, is_forward=is_forward),
            axis=a,
            is_forward=is_forward,
        )
        for a in range(-3, 0)
    ]
    return jnp.stack([dy(fz) - dz(fy), dz(fx) - dx(fz), dx(fy) - dy(fx)], axis=-4)


def source_waveform(
    t: ArrayLike,
    omegas: FreqSpace,
    rise_time: float,
) -> jax.Array:
    # NOTE: ``rise_time`` refers to the amount of time taken for the sigmoid
    # envelope function to go from an amplitude of roughly ``0.01`` to ``0.99``.

    # NOTE: This happens when we go from ``jax.nn.sigmoid(-4.6)`` to
    # ``jax.nn.sigmoid(+4.6)``.

    return (jax.nn.sigmoid(4.6 * (2 * t / rise_time - 1))) * jnp.sum(
        (jnp.exp(1j * (omegas.freqs[:, None] * t + omegas.phases[:, None]))), axis=0
    )


def simulate_new(
    e_field: ArrayLike,
    h_field: ArrayLike,
    epsilon: ArrayLike,
    sigma: ArrayLike,
    sources: Sequence[InputSpec],
    outputs: Sequence[OutputSpec],
    grid: Grid,
    start_step: int,
    num_steps: int,
) -> Tuple[jax.Array, jax.Array, Tuple[jax.Array, ...]]:

    state, outs = fdtd.simulate(
        dt=grid.dt,
        grid=grid.du,
        permittivity=epsilon,
        conductivity=sigma,
        source=fdtd.Source(
            offset=sources[0].offset,
            field=sources[0].field,
            waveform=sources[0].waveform,
        ),
        output_spec=fdtd.OutputSpec(
            offsets=[outputs[0].offset],
            shapes=[outputs[0].shape],
            start=outputs[0].range.start + 1,
            interval=outputs[0].range.interval,
            num=outputs[0].range.num,
        ),
        state=fdtd.State(
            step=start_step,
            e_field=e_field,
            h_field=h_field,
        ),
    )
    return state.e_field, state.h_field, outs


# def simulate(
#     e_field: ArrayLike,
#     h_field: ArrayLike,
#     epsilon: ArrayLike,
#     sigma: ArrayLike,
#     sources: Sequence[InputSpec],
#     outputs: Sequence[OutputSpec],
#     grid: Grid,
#     start_step: int,
#     num_steps: int,
# ) -> Tuple[jax.Array, jax.Array, Tuple[jax.Array, ...]]:
#
#     # Precomputed update coefficients
#     z = sigma * grid.dt / (2 * epsilon)
#     ca = (1 - z) / (1 + z)
#     cb = grid.dt / epsilon / (1 + z)
#
#     def step_fn(step, state):
#         e, h, outs = state
#
#         h = h - grid.dt * curl(e, grid, is_forward=True)
#
#         u = curl(h, grid, is_forward=False)
#         for source in sources:
#             u = source.inject(u, step)
#         e = ca * e + cb * u
#
#         outs = tuple(output.update(out, e, step) for out, output in zip(outs, outputs))
#
#         return e, h, outs
#
#     # Initialize output arrays.
#     outs = tuple(output.init_array() for output in outputs)
#
#     # Run the simulation.
#     return jax.lax.fori_loop(
#         lower=start_step,
#         upper=start_step + num_steps,
#         body_fun=step_fn,
#         init_val=(e_field, h_field, outs),
#     )
#
#
def freq_projection(
    fields: ArrayLike,
    omegas: FreqSpace,
    snapshots: SnapshotRange,
    dt: float,
) -> jax.Array:
    # Build ``P`` matrix.
    wt = dt * omegas.freqs[None, :] * (snapshots.arange()[:, None] + 0.5)
    P = jnp.concatenate([jnp.cos(wt), -jnp.sin(wt)], axis=1)
    # plt.figure()
    # plt.imshow(P, vmin=-1, vmax=+1)
    # plt.figure()
    # plt.imshow(jnp.linalg.inv(P), vmin=-1, vmax=+1)
    # plt.figure()
    # plt.imshow(jnp.abs(P.T @ P), vmin=-1, vmax=+1)
    # print(f"{P}")

    # Project out frequency components.
    res = jnp.einsum("ij,j...->i...", jnp.linalg.inv(P), fields)
    return res[: omegas.num] + 1j * res[omegas.num :]


def wave_equation_errors(
    fields: ArrayLike,
    omegas: FreqSpace,
    epsilon: ArrayLike,
    sigma: ArrayLike,
    source: ArrayLike,
    grid: Grid,
) -> jax.Array:
    w = jnp.expand_dims(omegas.freqs, axis=range(-4, 0))
    phi = jnp.expand_dims(omegas.phases, axis=range(-4, 0))
    err = (
        curl(curl(fields, grid, is_forward=True), grid, is_forward=False)
        - w**2 * (epsilon - 1j * sigma / w) * fields
        + 1j * w * source * jnp.exp(1j * phi)
    )
    return (
        jnp.sqrt(jnp.sum(jnp.abs(err) ** 2, axis=(1, 2, 3, 4)))
        / (omegas.freqs * jnp.linalg.norm(source))
    ), err


def solve(
    omegas: FreqSpace,
    epsilon: ArrayLike,
    sigma: ArrayLike,
    source: ArrayLike,
    grid: Grid,
    err_thresh: float,
    max_steps: int,
) -> Tuple[jax.Array, jax.Array, int, bool]:
    """Solution fields and error for the wave equation at ``omegas``."""

    # Steps to sample against.
    if omegas.num > 1:  # Adjust dt.
        n = int(jnp.floor(omegas.sampling_interval / grid.dt))
        dt = omegas.sampling_interval / (n + 1)
        sample_every_n_steps = n + 1
        grid = Grid(dt=dt, du=grid.du)
    else:
        sample_every_n_steps = int(round(omegas.sampling_interval / grid.dt))
    sample_steps = sample_every_n_steps * (2 * omegas.num - 1) + 1

    print(f"{sample_every_n_steps}")

    # Heuristic for determining the number of steps to simulate for, before
    # checking for the termination condition.

    # TODO: Think about how long to run the simulation.
    min_sim_steps_heuristic = sum(grid.shape())  # TODO: Move constant.

    # Actual number of steps to simulate.
    steps_per_sim = max(min_sim_steps_heuristic, sample_steps)
    # steps_per_sim = sample_steps

    # Reset ``max_steps`` for what the actual maximum number of steps will be.
    if max_steps % steps_per_sim != 0:
        max_steps = max_steps + steps_per_sim - (max_steps % steps_per_sim)

    # Form input for simulation.
    inputspec = InputSpec(
        offset=(0, 0, 0),
        field=source,
        waveform=source_waveform(
            t=grid.dt * jnp.arange(max_steps),
            omegas=omegas,
            rise_time=1,  # TODO: Put constant somewhere better.
        ),
    )

    # Initial values.
    e_field, h_field = 2 * [jnp.zeros_like(epsilon)]

    errs_hist = []  # TODO: Remove.

    for start_step in range(0, max_steps, steps_per_sim):
        outputspec = OutputSpec(
            offset=(0, 0, 0),
            shape=grid.shape(),
            range=SnapshotRange(
                start=start_step + steps_per_sim - sample_steps,
                interval=sample_every_n_steps,
                num=2 * omegas.num,
            ),
        )
        print(f"{outputspec.range}")

        # Run simulation.
        e_field, h_field, outs = simulate_new(
            e_field=e_field,
            h_field=h_field,
            epsilon=epsilon,
            sigma=sigma,
            sources=[inputspec],
            outputs=[outputspec],
            grid=grid,
            start_step=start_step,
            num_steps=steps_per_sim,
        )

        # Infer time-harmonic fields.
        freq_fields = freq_projection(outs[0], omegas, outputspec.range, grid.dt)

        # Compute error.
        errs, err_fields = wave_equation_errors(
            fields=freq_fields,
            omegas=omegas,
            epsilon=epsilon,
            sigma=sigma,
            source=source,
            grid=grid,
        )
        errs_hist.append(errs)
        # print(f"errs: {errs}")

        if jnp.max(errs) < err_thresh:
            return (
                freq_fields,
                errs,
                start_step + steps_per_sim,
                True,
                err_fields,
                errs_hist,
            )

    return freq_fields, errs, max_steps, False, err_fields, errs_hist
