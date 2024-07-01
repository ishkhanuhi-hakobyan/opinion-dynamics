from mfg.PeriodicEx31 import *
import jax.numpy as jnp
import numpy as np
from jax import vmap
import matplotlib.pyplot as plt
from jax.config import config
import munch

config.update("jax_enable_x64", True)
np.set_printoptions(precision=20)

cfg = munch.munchify({
    'T' : 1,
    'Nt': 50,
    'xl': 0,
    'xr': 1,
    'N' : 50,
    'nu': 1,
    'alpha': 1,
    'eps': 0.1,
    'hjb_epoch': 100,
    'hjb_lr': 1,
    'epoch': 100,
    'lr': 1,
    'tol' : 10 ** (-6),
})

#
# N = cfg.N
# xx = jnp.arange(0, N) / N
# h = 1/N
#
# solver = PeriodicEx31(cfg.T, cfg.Nt, cfg.N, cfg.nu, cfg.alpha, cfg.eps)


solver = PeriodicEx31(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.N, cfg.nu, cfg.alpha, cfg.eps)

TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
TT, XX = jnp.meshgrid(TT, XX)

# U, M = solver.solve_mfg(cfg.lr, cfg.tol, cfg.epoch)
U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)



TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(-1, 1, cfg.N, endpoint=False)
TT, XX = jnp.meshgrid(TT, XX)

fig = plt.figure()
plt.plot(XX, U[-1, :])
plt.xlabel(r"$x$")
plt.ylabel(r"$u(T)$")

fig = plt.figure()
plt.plot(XX, M[-1, :])
plt.xlabel(r"$x$")
plt.ylabel(r"$m(T)$")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(TT.T, XX.T, U, cmap='viridis')
fig.colorbar(surf, label=r"$u$", pad=0.1)
ax.set_title(r"$u$")
ax.set_xlabel(r"t")
ax.set_ylabel(r"x")
ax.set_zlabel(r"$u$")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TT.T, XX.T, M, cmap='viridis')
fig.colorbar(surf, label=r"$m$", pad=0.1)

ax.set_title(r"$m$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_zlabel(r"$m$")

plt.show()