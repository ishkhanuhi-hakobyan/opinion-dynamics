from mfg.PeriodicEx31_Local_Exp import *
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


solver = PeriodicEx31(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.N, cfg.nu, cfg.alpha, cfg.eps)

TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
TT, XX = jnp.meshgrid(TT, XX)

# U, M = solver.solve_mfg(cfg.lr, cfg.tol, cfg.epoch)
U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)


TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
TT, XX = jnp.meshgrid(TT, XX)

points = jnp.concatenate((TT.T.reshape((-1, 1)), XX.T.reshape((-1, 1))), axis=1)

def v(t, x):
    return t ** 2 / 2 + cfg.alpha * jnp.cos(2 * jnp.pi * x) / (2 * jnp.pi)


vs = vmap(v)(points[:, 0], points[:, 1])

###########################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(TT.T, XX.T, U, cmap='viridis')
fig.colorbar(surf, pad=0.1)
ax.set_title(r"$u$")
ax.set_xlabel(r"t")
ax.set_ylabel(r"x")
ax.set_zlabel(r"$u$")

###########################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TT.T, XX.T, jnp.reshape(vs, TT.T.shape), cmap='viridis')
fig.colorbar(surf, pad=0.1)

ax.set_title(r"$u^*$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_zlabel(r"$u^*$")

###########################################################################
fig = plt.figure()
ax = fig.add_subplot(111)
surf = ax.contourf(TT.T, XX.T, jnp.abs(U - jnp.reshape(vs, TT.T.shape)), cmap='viridis')
fig.colorbar(surf, pad=0.1)

ax.set_title(r"$\|u-u^*\|$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")

###########################################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TT.T, XX.T, M, cmap='viridis')
fig.colorbar(surf, pad=0.1)

ax.set_title(r"$m$")
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x$")
ax.set_zlabel(r"$m$")

plt.show()