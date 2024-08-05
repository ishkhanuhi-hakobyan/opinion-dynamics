import jax.numpy as jnp
from jax import vmap, jacfwd
from jax.scipy.stats import norm
import munch
from scipy.linalg import solve
from scipy.linalg import lstsq
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

x_d = 2
y_d = 2


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
        self.XX = jnp.linspace(xl, xr, N, endpoint=False)
        self.YY = jnp.linspace(yl, yr, M, endpoint=False)
        self.XX, self.YY = jnp.meshgrid(self.XX, self.YY)
        self.eps = eps

    def m0(self, x, y):
        x_midpoint = 1
        y_midpoint = 1
        sigma_mu = 1
        return norm.pdf(x, loc=x_midpoint, scale=sigma_mu) * norm.pdf(y, loc=y_midpoint, scale=sigma_mu)

    def uT(self, x, y):
        value = (x - x_d) ** 2 + (y - y_d) ** 2
        return value

    def g(self, q1, q2):
        return 0.5 * ((q1 ** 2) + (q2 ** 2))

    def hamilton(self, U):
        Ux = (jnp.roll(U, -1, axis=0) - jnp.roll(U, 1, axis=0)) / (2 * self.hx)
        Uy = (jnp.roll(U, -1, axis=1) - jnp.roll(U, 1, axis=1)) / (2 * self.hy)

        Hamiltonian = vmap(lambda qx, qy: self.g(qx, qy))(Ux, Uy)
        return Hamiltonian

    def fp_linearized_part(self, U, M):
        UR = jnp.roll(U, -1, axis=0)  # U shifted left (i+1)
        UL = jnp.roll(U, 1, axis=0)  # U shifted right (i-1)

        UR_y = jnp.roll(U, -1, axis=1)  # U shifted up (j+1)
        UL_y = jnp.roll(U, 1, axis=1)  # U shifted down (j-1)

        MR = jnp.roll(M, -1, axis=0)  # M shifted left (i+1)
        ML = jnp.roll(M, 1, axis=0)  # M shifted right (i-1)

        MR_y = jnp.roll(M, -1, axis=1)  # M shifted up (j+1)
        ML_y = jnp.roll(M, 1, axis=1)  # M shifted down (j-1)

        u_ip1_jp1 = jnp.roll(jnp.roll(U, -1, axis=0), -1, axis=1)  # u_{i+1,j+1}
        u_ip1_jm1 = jnp.roll(jnp.roll(U, -1, axis=0), 1, axis=1)  # u_{i+1,j-1}
        u_im1_jp1 = jnp.roll(jnp.roll(U, 1, axis=0), -1, axis=1)  # u_{i-1,j+1}
        u_im1_jm1 = jnp.roll(jnp.roll(U, 1, axis=0), 1, axis=1)  # u_{i-1,j-1}

        u_xy = (u_ip1_jp1 - u_ip1_jm1 - u_im1_jp1 + u_im1_jm1) / (4 * self.hx * self.hy)

        Delta_U = - (4 * U - UR - UL - UR_y - UL_y) / (self.hx ** 2)

        Ux = (UR - UL) / (2 * self.hx)
        Uy = (UR_y - UL_y) / (2 * self.hy)

        Mx = (MR - ML) / (2 * self.hx)
        My = (MR_y - ML_y) / (2 * self.hy)

        # Ensure that the multiplication doesn't lead to NaN or inf by using jnp.clip
        val = -(jnp.multiply(M, Delta_U) +
                2 * jnp.multiply(M, u_xy) +
                jnp.multiply((Mx + My), (Ux + Uy)))

        return val

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
        M0 = vmap(self.m0)(self.XX, self.YY)
        UT = vmap(self.uT)(self.XX, self.YY)

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
    'T': 1,
    'Nt': 10,
    'xl': -5,
    'xr': 5,
    'yl': -5,
    'yr': 5,
    'N': 30,
    'M': 30,
    'nu': 1,
    'alpha': 1,
    'eps': 1,
    'hjb_epoch': 100,
    'hjb_lr': 1,
    'epoch': 100,
    'lr': 1,
    'tol': 10 ** (-8),
})

solver = OnePopulationalMFG(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.yl, cfg.yr, cfg.N, cfg.M, cfg.nu, cfg.alpha, cfg.eps)

TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
YY = jnp.linspace(cfg.yl, cfg.yr, cfg.M, endpoint=False)
TT, XX, YY = jnp.meshgrid(TT, XX, YY)

# Solve to get U and M
U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)

# Create the 3D plot for m0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX[:, -1, :], YY[:, -1, :], M[0, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$m_0(x, y)$")
ax.set_title("Initial Distribution $m_0(x, y)$")

ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5

plt.tight_layout()  # Adjust layout to fit all elements
plt.show()


# Create the 3D plot for m0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX[:, -1, :], YY[:, -1, :], U[-1, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$m_0(x, y)$")
ax.set_title("Final Distribution $m_T(x, y)$")
z_min, z_max = ax.get_zlim()  # Get the current z-axis limits
ax.plot([x_d, x_d], [y_d, y_d], [z_min, z_max], color='green', linestyle='--', label=r"Line at $(x_{d}, y_{d})$")
ax.legend()

plt.show()

# Create the 3D plot for m0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX[:, -1, :], YY[:, -1, :], M[-1, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$m_0(x, y)$")
ax.set_title("Final Distribution $m_T(x, y)$")
z_min, z_max = ax.get_zlim()  # Get the current z-axis limits
ax.plot([x_d, x_d], [y_d, y_d], [z_min, z_max], color='green', linestyle='--', label=r"Line at $(x_{d}, y_{d})$")
ax.legend()

plt.show()

# Set a fixed aspect ratio
ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5

plt.tight_layout()  # Adjust layout to fit all elements
plt.show()

# Define time steps to visualize
time_steps = range(cfg.Nt + 1)  # Visualize all time steps

z_min = jnp.min(M)
z_max = jnp.max(M)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


def update_plot(t):
    ax.clear()
    surf = ax.plot_surface(XX[:, -1, :], YY[:, -1, :], M[t, :, :], cmap=cm.coolwarm, alpha=0.8)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$m(t, x, y)$")
    ax.set_title(f"Distribution at t = {t}")

    # Apply consistent z-axis limits
    ax.set_zlim(z_min, z_max)

    # Plot the vertical line
    ax.plot([x_d, x_d], [y_d, y_d], [z_min, z_max], color='green', linestyle='--', label=r"Line at $(x_{d}, y_{d})$")
    ax.legend()

    # Set a fixed aspect ratio
    ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5


ani = FuncAnimation(fig, update_plot, frames=range(cfg.Nt + 1), repeat=False)

# To save the animation as a GIF file
ani.save('distribution_evolution.gif', writer='imagemagick', dpi=80)

plt.show()