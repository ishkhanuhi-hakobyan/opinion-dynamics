import jax.numpy as jnp
from jax import vmap, jacfwd
from jax.scipy.stats import norm
import munch
import numpy as np
from scipy.linalg import solve
from scipy.linalg import lstsq

import matplotlib.pyplot as plt

x_d = 0
y_d = -2

class OnePopulationalMFG(object):
    def __init__(self, T, Nt, xl, xr, yl, yr, N, M, nu, alpha, eps):
        """
        Input:
        T:          float, the terminal time
        Nt:         integer, the number of time intervals
        xl, xr:     float, the left and right boundary of the interval for x
        yl, yr:     float, the left and right boundary of the interval for y
        N, M:       integer, the number of intervals for x and y
        nu:         float, the viscosity constant
        alpha:      float, the coefficient of the susceptibility of the population
        eps:        float, define the closeness of agents
        """
        self.T = T
        self.Nt = Nt
        self.dt = T / Nt
        self.xl = xl
        self.xr = xr
        self.yl = yl
        self.yr = yr
        self.N = N
        self.M = M
        self.nu = nu
        self.alpha = alpha
        self.hx = (xr - xl) / N
        self.hy = (yr - yl) / M
        self.X = jnp.linspace(xl, xr, N, endpoint=False)
        self.Y = jnp.linspace(yl, yr, M, endpoint=False)
        self.eps = eps

    def m0(self, x, y):
        x_midpoint = -2  # Start away from the desired point
        y_midpoint = 2
        sigma_mu = 1  # Define your standard deviation
        return norm.pdf(x, loc=x_midpoint, scale=sigma_mu) * norm.pdf(y, loc=y_midpoint, scale=sigma_mu)

    def uT(self, x, y):
        return (x - x_d) ** 2 + (y - y_d) ** 2

    def g(self, q1, q2):
        p1 = jnp.minimum(q1, 0)
        p2 = jnp.maximum(q2, 0)
        return 0.5 * ((p1 ** 2) + (p2 ** 2))

    def hamilton(self, U):
        dUx = (jnp.roll(U, -1, axis=0) - jnp.roll(U, 1, axis=0)) / (2 * self.hx)
        dUy = (jnp.roll(U, -1, axis=1) - jnp.roll(U, 1, axis=1)) / (2 * self.hy)

        Hamiltonian = vmap(lambda qx, qy: self.g(qx, qy))(dUx, dUy)
        return Hamiltonian

    def fp_linearized_part(self, U, M):
        UR = (jnp.roll(U, -1, axis=0) - U) / self.hx
        UL = (U - jnp.roll(U, 1, axis=0)) / self.hx

        UR_flatten = UR.flatten()
        UL_flatten = UL.flatten()
        M_flatten = M.flatten()

        dGq1 = lambda q1, q2: jnp.minimum(q1, 0)
        dGq2 = lambda q1, q2: jnp.maximum(q2, 0)

        dGq1s = vmap(dGq1)(UR_flatten, UL_flatten)
        dGq2s = vmap(dGq2)(UR_flatten, UL_flatten)

        dGq1s = jnp.multiply(dGq1s, M_flatten)
        dGq2s = jnp.multiply(dGq2s, M_flatten)

        dGqDifference1 = dGq1s - dGq2s

        dGqs2R = jnp.roll(dGq2s, -1).reshape(UR.shape)
        dGqs1L = jnp.roll(dGq1s, 1).reshape(UR.shape)

        A = - (dGqDifference1 + dGqs2R.flatten() - dGqs1L.flatten()) / self.hx

        return A.reshape(U.shape)

    def hjb(self, t, Uk1, Uk, Mk1):
        UR = jnp.roll(Uk, -1, axis=0)
        UL = jnp.roll(Uk, 1, axis=0)
        UR_y = jnp.roll(Uk, -1, axis=1)
        UL_y = jnp.roll(Uk, 1, axis=1)

        Delta_U = - (4 * Uk - UR - UL - UR_y - UL_y) / (self.hx ** 2)

        Dt_U = (Uk1 - Uk) / self.dt

        Hamiltonian = self.hamilton(Uk)

        return -Dt_U - self.nu * Delta_U + Hamiltonian - Mk1**2

    def fp(self, Uk, Mk1, Mk):
        Dt_M = (Mk1 - Mk) / self.dt

        Mk1R = jnp.roll(Mk1, -1, axis=0)
        Mk1L = jnp.roll(Mk1, 1, axis=0)
        Mk1R_y = jnp.roll(Mk1, -1, axis=1)
        Mk1L_y = jnp.roll(Mk1, 1, axis=1)

        Delta_M = - (4 * Mk1 - Mk1R - Mk1L - Mk1R_y - Mk1L_y) / (self.hx ** 2)

        adj = self.fp_linearized_part(Uk, Mk1)

        return Dt_M - self.nu * Delta_M + adj

    def hjb_sys(self, U, M):
        ts = jnp.linspace(0, self.T, self.Nt + 1)
        hjbs = vmap(self.hjb)(ts[:-1], U[1:, :], U[:-1, :], M[1:, :])

        return hjbs

    def fp_sys(self, U, M):
        fps = vmap(self.fp)(U[:-1, :], M[1:, :], M[:-1, :])

        return fps

    def prolong(self, Uvec, Mvec):
        Umtx = jnp.reshape(Uvec, (self.Nt, self.N, self.M))
        Mmtx = jnp.reshape(Mvec, (self.Nt, self.N, self.M))

        U = jnp.zeros((self.Nt + 1, self.N, self.M))
        M = jnp.zeros((self.Nt + 1, self.N, self.M))

        U = U.at[:self.Nt, :, :].set(Umtx)
        M = M.at[1:, :, :].set(Mmtx)

        # compute the values of m at time 0 and the values of u at time T
        M0 = vmap(self.m0)(self.X, self.Y)
        UT = vmap(self.uT)(self.X, self.Y)

        U = U.at[self.Nt, :, :].set(UT)
        M = M.at[0, :, :].set(M0)
        return U, M

    def hjb_sys_vec(self, Uvec, Mvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.hjb_sys(U, M).flatten()

    def fp_sys_vec(self, Mvec, Uvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.fp_sys(U, M).flatten()

    def solve_hjb(self, U0, M, lr, tol=10**(-6), epoch=100):
        error = 1
        iter_num = 0

        U = U0
        while error > tol and iter_num < epoch:
            b = self.hjb_sys_vec(U, M)
            jacobi = jacfwd(self.hjb_sys_vec)(U, M)
            dz = solve(jacobi, b.flatten())
            U = U - lr * dz

            error = jnp.dot(dz, dz)
            print("             the error of solving hjb is {}".format(error))

            iter_num = iter_num + 1

        return U

    def solve_fp(self, M0, U):
        jacobi = jacfwd(self.fp_sys_vec)(M0, U)
        b = self.fp_sys_vec(jnp.zeros(len(jacobi)), U)
        M = lstsq(jacobi, -b)
        return M[0]

    def solve(self, tol=10**(-6), epoch=100, hjb_lr = 1, hjb_epoch = 100):
        U = jnp.zeros((self.Nt, self.N, self.M)).flatten()
        M = jnp.zeros((self.Nt, self.N, self.M)).flatten()
        error = 1
        iter_num = 0
        while error > tol and iter_num < epoch:
            U1 = self.solve_hjb(U, M, hjb_lr, epoch=hjb_epoch)
            M1 = self.solve_fp(M, U1)

            Uerr = U1 - U
            Merr = M1 - M

            error = jnp.dot(Uerr, Uerr) + jnp.dot(Merr, Merr)
            print('the mfg error is {}'.format(error))

            iter_num = iter_num + 1

            U = U1
            M = M1

        U, M = self.prolong(U, M)
        return U, M


cfg = munch.munchify({
    'T' : 1,
    'Nt': 20,
    'xl': -5,
    'xr': 5,
    'yl': -5,
    'yr': 5,
    'N': 20,
    'M': 20,
    'nu': 1,
    'alpha': 1,
    'eps': 1,
    'hjb_epoch': 100,
    'hjb_lr': 1,
    'epoch': 100,
    'lr': 1,
    'tol' : 10 ** (-7),
})

solver = OnePopulationalMFG(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.yl, cfg.yr, cfg.N, cfg.M, cfg.nu, cfg.alpha, cfg.eps)

TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
YY = jnp.linspace(cfg.yl, cfg.yr, cfg.M, endpoint=False)
TT, XX, YY = jnp.meshgrid(TT, XX, YY)

# Solve to get U and M
U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)

# Select the final time step and reshape U and M
time_step = -1  # Last time step

U_final = U[time_step].reshape(cfg.N, cfg.M)
M_final = M[time_step].reshape(cfg.N, cfg.M)

# Generate the meshgrid for XX and YY (ignore TT for plotting)
XX_grid, YY_grid = np.meshgrid(XX[:, 0, 0], YY[0, 0, :])

# Plot U
fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(XX_grid, YY_grid, U_final, cmap='viridis')
ax.set_title('Potential Function U at Final Time Step')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('U')

# Plot M
ax = fig.add_subplot(122, projection='3d')
ax.plot_surface(XX_grid, YY_grid, M_final, cmap='viridis')
ax.set_title('Density Function M at Final Time Step')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('M')

plt.tight_layout()
plt.show()

# Generate a consistent meshgrid that matches the shape of U_final and M_final
XX_2D, YY_2D = np.meshgrid(XX[:cfg.N], YY[:cfg.M])

# Extract the final time step for U and M, ensuring they are 2D
U_final = U[-1, :, :]
M_final = M[-1, :, :]

