from functools import partial

import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as onp
from scipy.optimize import brentq

import hyperwave.fdtd as fdtd
from hyperwave.typing import Range, Subfield, Volume


# 1. Define the properties of the waveguide
width = 0.5
height = 0.22
n_clad = 1.45 # SiO2 substrate
n_wg = 3.44   # Si waveguide
wl = 1.55

# 2. Define the simulation window size
Lx = 5 * width
Lz = 5 * height
Ly = 4.0
dL = 50e-3

# 3. Define the geometry - SIMPLE EXAMPLE AKA NO AVERAGING
Nx, Ny, Nz = [int(Ll // dL) for Ll in (Lx, Ly, Lz)]

eps = jnp.ones((3, Nx, Ny, Nz)) * n_clad ** 2
xspacing = int((Lx - width) / 2 // dL)
zspacing = int((Lz - height) / 2 // dL)
eps = eps.at[:, xspacing:-xspacing, :, zspacing:-zspacing].set(n_wg ** 2)


# 4. Define the coordinates
xx = jnp.linspace(-Lx/2, Lx/2-dL, Nx)
yy = jnp.linspace(-Ly/2, Ly/2-dL, Ny)
zz = jnp.linspace(-Lz/2, Lz/2-dL, Nz)
grid = (
    jnp.array((xx, xx + dL/2)).T,
    jnp.array((yy, yy + dL/2)).T,
    jnp.array((zz, zz + dL/2)).T,
)

fig, axes = plt.subplots(1, 3)
extent_xz = [min(xx), max(xx), min(zz), max(zz)]
extent_yz = [min(yy), max(yy), min(zz), max(zz)]
extent_xy = [min(xx), max(xx), min(yy), max(yy)]

axes[0].imshow(eps[0, :, Ny // 2, :].T, extent=extent_xz, cmap='Greys')
axes[0].set_xlabel("X coordinates [$um$]")
axes[0].set_ylabel("Z coordinates [$um$]")

axes[1].imshow(eps[0, Nx//2, :, :].T, extent=extent_yz, cmap='Greys')
axes[1].set_xlabel("Y coordinates [$um$]")
axes[1].set_ylabel("Z coordinates [$um$]")

axes[2].imshow(eps[0, :, :, Nz // 2].T, extent=extent_xy, cmap='Greys')
axes[2].set_xlabel("X coordinates [$um$]")
axes[2].set_ylabel("Y coordinates [$um$]")

fig.tight_layout()
plt.title("Permittivity")
plt.savefig('indexes.png')
plt.close()

# 5. Source calculation - Exact profile
# From Yariv "Optical Electronics in Modern Communications", Chapter 3
# The mode is solved using the hypothesis of separable equation
# i.e. it will be solve for the two cross_sections
# h = jnp.sqrt(jnp.square(n_wg * k0) - )
k0 = 2 * jnp.pi / wl

def get_mode(k0, nclad, ncore, d, xx, N, idw_min):
    V2 = lambda w: (ncore**2 - nclad**2) * (w * k0 / 2)**2
    trans_eq = lambda u, V: V - jnp.square(u) - jnp.square(u * jnp.tan(u))

    V2x = V2(d)
    ux = brentq(lambda x: trans_eq(x, V2x), 0, jnp.pi/2)
    kx = 2 * ux / d
    neffx = onp.sqrt(-kx**2 + (n_wg * k0)**2) / k0

    alpha = kx * jnp.tan(kx * d / 2)
    C = jnp.cos(kx * d / 2) / jnp.exp(-alpha * d / 2)
    D = C

    E_mode = jnp.ones((N,))
    E_mode = E_mode.at[:idw_min].set(D * jnp.exp(alpha * xx[:idw_min]))
    E_mode = E_mode.at[idw_min:-idw_min].set(jnp.cos(kx * xx[idw_min:-idw_min]))
    E_mode = E_mode.at[-idw_min:].set(C * jnp.exp(-alpha * xx[-idw_min:]))

    return E_mode

mode_x = get_mode(k0, n_clad, n_wg, width, xx, Nx, xspacing)
mode_z = get_mode(k0, n_clad, n_wg, height, zz, Nz, zspacing)

mode_full_x = jnp.repeat(jnp.expand_dims(mode_x, 1), Nz, axis=1)
mode_full_z = jnp.repeat(jnp.expand_dims(mode_z, 0), Nx, axis=0)

mode = jnp.ones((3, Nx, 1, Nz))
# Add the mode as Ez at 1/4 of the window
mode = mode.at[-1, :, 0 , :].set(mode_full_x * mode_full_z)
src_ypos = Ny // 4
mode_src= Subfield((0, src_ypos, 0,), mode)

# plt.imshow(mode_full_x * mode_full_z)
plt.imshow(mode[2].squeeze().T, extent=extent_xz, cmap='inferno')
plt.xlabel("X coordinates [$um$]")
plt.ylabel("Z coordinates [$um$]")
plt.title("Computed mode")
plt.savefig("mode.png")
plt.close()

# 6. Boundary conditions calculation
conductivity = jnp.zeros_like(eps)

# 7. Simulate
dt = 0.99 * jnp.min(eps) / jnp.sqrt(3)
# tmax = 1e-11
Ntmax = 4
tt = jnp.arange(0, Ntmax) * dt

outp_volume1 = Volume(offset=(0, src_ypos + 2, 0), shape=(Nx, 1, Nz))
outp_volume2 = Volume(offset=(0, Ny - src_ypos, 0), shape=(Nx, 1, Nz))
outp_range = Range(1, 100, int(tt[-1] // dt))

mode_temp = jnp.exp(-jnp.square((tt - len(tt)* 2/5) / 1e4))

# plt.plot(tt, jnp.abs(mode_temp))
# plt.show()
state, outs = fdtd.simulate(
    dt=dt,
    grid=grid,
    permittivity=eps,
    conductivity=conductivity,
    source_field=mode_src,
    source_waveform=mode_temp,
    output_volumes=(outp_volume1, outp_volume2),
    snapshot_range=outp_range,
)

# 8. Plot results
fig, axes = plt.subplots(2, 3)
for idax in range(len(axes.flatten())):
    iout, idir = idax // 3, idax % 3
    axes[iout,idir].imshow(
        outs[iout][-1].squeeze()[idir],
        extent=extent_xz,
    )
plt.show()
