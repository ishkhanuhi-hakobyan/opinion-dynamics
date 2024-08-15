import torch
import torch.nn.functional as F
import numpy as np
import munch
from scipy.linalg import solve, lstsq
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


x_d = 2
y_d = 2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

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
        self.XX = torch.linspace(xl, xr, N, device=device)
        self.YY = torch.linspace(yl, yr, M, device=device)
        self.XX, self.YY = torch.meshgrid(self.XX, self.YY)
        self.eps = eps

    def m0(self, x, y):
        x_midpoint = 1
        y_midpoint = 1
        sigma_mu = 1
        return torch.exp(-0.5 * ((x - x_midpoint)**2 + (y - y_midpoint)**2) / sigma_mu**2) / (2 * np.pi * sigma_mu**2)

    def uT(self, x, y):
        value = (x - x_d) ** 2 + (y - y_d) ** 2
        return value

    def g(self, q1, q2):
        return 0.5 * (q1 ** 2 + q2 ** 2)

    def hamilton(self, U):
        Ux = (torch.roll(U, shifts=-1, dims=0) - torch.roll(U, shifts=1, dims=0)) / (2 * self.hx)
        Uy = (torch.roll(U, shifts=-1, dims=1) - torch.roll(U, shifts=1, dims=1)) / (2 * self.hy)

        Hamiltonian = self.g(Ux, Uy)
        return Hamiltonian

    def fp_linearized_part(self, U, M):
        UR = torch.roll(U, shifts=-1, dims=0)  # U shifted left (i+1)
        UL = torch.roll(U, shifts=1, dims=0)  # U shifted right (i-1)

        UR_y = torch.roll(U, shifts=-1, dims=1)  # U shifted up (j+1)
        UL_y = torch.roll(U, shifts=1, dims=1)  # U shifted down (j-1)

        MR = torch.roll(M, shifts=-1, dims=0)  # M shifted left (i+1)
        ML = torch.roll(M, shifts=1, dims=0)  # M shifted right (i-1)

        MR_y = torch.roll(M, shifts=-1, dims=1)  # M shifted up (j+1)
        ML_y = torch.roll(M, shifts=1, dims=1)  # M shifted down (j-1)

        Delta_U = (-2 * U + UR + UL) / (self.hx ** 2) + (-2 * U + UR_y + UL_y) / (self.hy ** 2)

        Ux = (UR - UL) / (2 * self.hx)
        Uy = (UR_y - UL_y) / (2 * self.hy)

        Mx = (MR - ML) / (2 * self.hx)
        My = (MR_y - ML_y) / (2 * self.hy)

        val = -(M * Delta_U + Ux * Mx + Uy * My)
        return val

    def hjb(self, t, Uk1, Uk, Mk1):
        UR = torch.roll(Uk, shifts=-1, dims=0)
        UL = torch.roll(Uk, shifts=1, dims=0)
        UR_y = torch.roll(Uk, shifts=-1, dims=1)
        UL_y = torch.roll(Uk, shifts=1, dims=1)

        Delta_U = - (2 * Uk - UR - UL) / (self.hx ** 2) - (2 * Uk - UR_y - UL_y) / (self.hy ** 2)

        Dt_U = (Uk1 - Uk) / self.dt

        Hamiltonian = self.hamilton(Uk)

        return -Dt_U - self.nu * Delta_U + Hamiltonian - Mk1**2

    def fp(self, Uk, Mk1, Mk):
        Dt_M = (Mk1 - Mk) / self.dt

        Mk1R = torch.roll(Mk1, shifts=-1, dims=0)
        Mk1L = torch.roll(Mk1, shifts=1, dims=0)
        Mk1R_y = torch.roll(Mk1, shifts=-1, dims=1)
        Mk1L_y = torch.roll(Mk1, shifts=1, dims=1)

        Delta_M = - (2 * Mk1 - Mk1R - Mk1L) / (self.hx ** 2) - (2 * Mk1 - Mk1R_y - Mk1L_y) / (self.hy ** 2)
        adj = self.fp_linearized_part(Uk, Mk1)

        return Dt_M - self.nu * Delta_M + adj

    def hjb_sys(self, U, M):
        ts = torch.linspace(0, self.T, self.Nt + 1, device=device)
        hjbs = torch.stack([self.hjb(t, U[i+1], U[i], M[i+1]) for i, t in enumerate(ts[:-1])])

        return hjbs

    def fp_sys(self, U, M):
        fps = torch.stack([self.fp(U[i], M[i+1], M[i]) for i in range(len(U)-1)])

        return fps

    def prolong(self, Uvec, Mvec):
        Umtx = Uvec.view(self.Nt, self.N, self.M)
        Mmtx = Mvec.view(self.Nt, self.N, self.M)

        U = torch.zeros((self.Nt + 1, self.N, self.M), device=device)
        M = torch.zeros((self.Nt + 1, self.N, self.M), device=device)

        U[:self.Nt] = Umtx
        M[1:] = Mmtx

        # compute the values of m at time 0 and the values of u at time T
        M0 = self.m0(self.XX, self.YY)
        UT = self.uT(self.XX, self.YY)

        U[self.Nt] = UT
        M[0] = M0
        return U, M

    def hjb_sys_vec(self, Uvec, Mvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.hjb_sys(U, M).flatten()

    def fp_sys_vec(self, Mvec, Uvec):
        U, M = self.prolong(Uvec, Mvec)
        return self.fp_sys(U, M).flatten()

    def solve_hjb(self, U0, M, lr, tol=1e-6, epoch=100):
        error = 1
        iter_num = 0

        U = U0
        while error > tol and iter_num < epoch:
            b = self.hjb_sys_vec(U, M)
            jacobi = torch.autograd.functional.jacobian(lambda u: self.hjb_sys_vec(u, M), U)
            dz = torch.linalg.solve(jacobi, b.unsqueeze(-1))
            U = U - lr * dz.squeeze()

            error = torch.dot(dz.squeeze(), dz.squeeze()).item()
            print(f"             the error of solving hjb is {error}")

            iter_num += 1

        return U

    def solve_fp(self, M0, U):
        jacobi = torch.autograd.functional.jacobian(lambda m: self.fp_sys_vec(m, U), M0)
        b = self.fp_sys_vec(torch.zeros_like(M0), U)
        M, _ = torch.solve(-b.unsqueeze(-1), jacobi)
        return M.squeeze()

    def solve(self, tol=1e-6, epoch=100, hjb_lr=1, hjb_epoch=100):
        U = torch.zeros((self.Nt, self.N, self.M), device=device).flatten()
        M = torch.zeros((self.Nt, self.N, self.M), device=device).flatten()
        error = 1
        iter_num = 0
        while error > tol and iter_num < epoch:
            U1 = self.solve_hjb(U, M, hjb_lr, epoch=hjb_epoch)
            M1 = self.solve_fp(M, U1)
            Uerr = U1 - U
            Merr = M1 - M
            error = (Uerr.norm() ** 2 + Merr.norm() ** 2).item()
            print(f'the mfg error is {error}')

            U = U1
            M = M1

        U, M = self.prolong(U, M)
        return U, M

cfg = munch.munchify({
    'T': 1,
    'Nt': 10,
    'xl': -6,
    'xr': 6,
    'yl': -6,
    'yr': 6,
    'N': 30,
    'M': 30,
    'nu': 1,
    'alpha': 1,
    'eps': 1,
    'hjb_epoch': 100,
    'hjb_lr': 1,
    'epoch': 100,
    'lr': 0.8,
    'tol': 10 ** (-7),
})

solver = OnePopulationalMFG(cfg.T, cfg.Nt, cfg.xl, cfg.xr, cfg.yl, cfg.yr, cfg.N, cfg.M, cfg.nu, cfg.alpha, cfg.eps)

TT = torch.linspace(0, cfg.T, cfg.Nt + 1, device=device)
XX = torch.linspace(cfg.xl, cfg.xr, cfg.N, device=device)
YY = torch.linspace(cfg.yl, cfg.yr, cfg.M, device=device)
TT, XX, YY = torch.meshgrid(TT, XX, YY)


# Solve to get U and M
U, M = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)

# Convert PyTorch tensors to NumPy arrays for plotting
U = U.cpu().numpy()
M = M.cpu().numpy()
XX_np = XX.cpu().numpy()
YY_np = YY.cpu().numpy()

# Create the 3D plot for m0
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX_np[:, -1, :], YY_np[:, -1, :], M[0, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$m_0(x, y)$")
ax.set_title("Initial Distribution $m_0(x, y)$")

ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5

plt.tight_layout()  # Adjust layout to fit all elements
plt.show()

# Create the 3D plot for uT
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX_np[:, -1, :], YY_np[:, -1, :], U[-1, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$u_T(x, y)$")
ax.set_title("Final Distribution $u_T(x, y)$")
z_min, z_max = ax.get_zlim()  # Get the current z-axis limits
ax.plot([x_d, x_d], [y_d, y_d], [z_min, z_max], color='green', linestyle='--', label=r"Line at $(x_{d}, y_{d})$")
ax.legend()

plt.show()

# Create the 3D plot for mT
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(XX_np[:, -1, :], YY_np[:, -1, :], M[-1, :, :], cmap=cm.coolwarm, alpha=0.8)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_zlabel(r"$m_T(x, y)$")
ax.set_title("Final Distribution $m_T(x, y)$")
z_min, z_max = ax.get_zlim()  # Get the current z-axis limits
ax.plot([x_d, x_d], [y_d, y_d], [z_min, z_max], color='green', linestyle='--', label=r"Line at $(x_{d}, y_{d})$")
ax.legend()

plt.show()

# Set a fixed aspect ratio
ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio is 1:1:0.5

plt.tight_layout()  # Adjust layout to fit all elements
plt.show()

time_steps = range(cfg.Nt + 1)  # Visualize all time steps

z_min = M.min()
z_max = M.max()

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


# Find the indices of the maximum value in M[t, :, :]
def find_peak(t):
    # Flatten the arrays and find the peak in the flattened data
    M_flat = M[t, :, :].flatten()
    XX_flat = XX_np[:, -1, :].flatten()
    YY_flat = YY_np[:, -1, :].flatten()

    max_idx = np.argmax(M_flat)

    x_peak = XX_flat[max_idx]
    y_peak = YY_flat[max_idx]
    z_peak = M_flat[max_idx]

    # Debugging: Print the peak values
    print(f"Peak coordinates at t={t}: x={x_peak}, y={y_peak}, z={z_peak}")

    return x_peak, y_peak, z_peak


def update_plot(t):
    ax.clear()
    surf = ax.plot_surface(XX_np[:, -1, :], YY_np[:, -1, :], M[t, :, :], cmap=cm.coolwarm, alpha=0.8)
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

    # Find and plot the peak
    x_peak, y_peak, z_peak = find_peak(t)
    ax.scatter([x_peak], [y_peak], [z_peak], color='red', s=50, label="Peak")

    # Annotate the peak values on the plot
    ax.text(x_peak, y_peak, z_peak, f"({x_peak:.2f}, {y_peak:.2f}, {z_peak:.2f})",
            color='black', fontsize=10, ha='center', va='bottom', weight='bold')

    ax.legend()

    # Optionally, print the coordinates of the peak
    print(f"At t={t}: Peak at x={x_peak}, y={y_peak}, z={z_peak}")


ani = FuncAnimation(fig, update_plot, frames=range(cfg.Nt + 1), repeat=False)

ani.save('plot2.gif', writer='pillow', dpi=80)

plt.show()