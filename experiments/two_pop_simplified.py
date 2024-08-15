import jax.numpy as jnp
from jax import jacfwd, vmap, jit
from jax.scipy.stats import norm
import matplotlib.pyplot as plt
import munch
import numpy as np

x_d1 = -1.5
x_d2 = 1


class TwoPopSimplified(object):
    def __init__(self, T, Nt, xl, xr, N, nu, alphas, sigmas, lambdas, eps):
        self.T = T
        self.Nt = Nt
        self.xl = xl
        self.xr = xr
        self.N = N
        self.nu = nu
        self.alphas = alphas
        self.sigmas = sigmas
        self.lambdas = lambdas
        self.dt = T / Nt
        self.dx = (xr - xl) / N
        self.x_grid = jnp.linspace(xl, xr, N)
        self.t_grid = jnp.linspace(0, T, Nt)
        self.eps = eps
        self.h = (xr - xl) / N

    def mu0(self, x,population_index):
        if population_index == 0:
            # return jnp.ones(x.shape) / (self.xr - self.xl)
            midpoint = (self.xr + self.xl) / 2
            sigma_mu = 1  # Define your standard deviation
            return norm.pdf(x, loc=midpoint, scale=sigma_mu)
        elif population_index == 1:
            midpoint = (self.xr + self.xl) / 2
            sigma_mu = 1  # Define your standard deviation
            return norm.pdf(x, loc=midpoint, scale=sigma_mu)
            # return jnp.ones(x.shape) / (self.xr - self.xl)

    def uT(self, x, population_index):
        if population_index == 0:
            return (x - x_d1) ** 2  # For population 1
        elif population_index == 1:
            return (x - x_d2) ** 2  # For population 2

    def prolong(self, Uvec, Mvec, idx):
        Umtx = jnp.reshape(Uvec, (self.Nt, self.N))
        Mmtx = jnp.reshape(Mvec, (self.Nt, self.N))

        U = jnp.zeros((self.Nt + 1, self.N))
        M = jnp.zeros((self.Nt + 1, self.N))

        U = U.at[:self.Nt, :].set(Umtx)
        M = M.at[1:, :].set(Mmtx)

        M0 = self.mu0(self.x_grid, idx)  # Assuming mu0 returns an array of shape (N,)
        UT = vmap(self.uT, in_axes=(0, None))(self.x_grid, idx)
        U = U.at[self.Nt, :].set(UT)
        M = M.at[0, :].set(M0)
        return U, M

    def g(self, q):
        return 0.5 * q ** 2

    def hamilton(self, U):
        dU = (jnp.roll(U, -1) - jnp.roll(U, 1)) / (2 * self.h)  # central difference approximation

        Hamiltonian = vmap(lambda q: self.g(q))(dU)

        return Hamiltonian

    def hjb(self, ts, Uk1, Uk, Mk1, idx):
        UR = jnp.roll(Uk, -1)
        UL = jnp.roll(Uk, 1)

        Delta_U = - (2 * Uk - UR - UL) / self.h ** 2

        Dt_U = (Uk1 - Uk) / self.dt

        Hamiltonian = self.hamilton(Uk)

        return -Dt_U - self.nu * Delta_U + Hamiltonian - Mk1**2

    def fp_linearized_part(self, U, M, idx):
        UR = (jnp.roll(U, -1) - U) / self.h
        UL = (U - jnp.roll(U, 1)) / self.h

        URF = UR.flatten()
        ULF = UL.flatten()

        # For the new Hamiltonian, we consider U directly
        dGq = lambda q: q  # derivative of 0.5 * q^2 is q

        dGqsR = vmap(dGq)(URF)
        dGqsL = vmap(dGq)(ULF)

        # Multiply by M as needed
        dGqsR = jnp.multiply(dGqsR, M.flatten())
        dGqsL = jnp.multiply(dGqsL, M.flatten())

        dGqDifference = dGqsR - dGqsL

        dGqsR_shifted = jnp.roll(dGqsR, -1)
        dGqsL_shifted = jnp.roll(dGqsL, 1)

        A = - (dGqDifference + dGqsR_shifted - dGqsL_shifted) / self.h

        # Since G_m is not used, remove related code and simplify
        MEQByHands = A  # Only A remains as B is not needed

        return MEQByHands

    def solve_hjb(self, U0, M, lr, idx, tol=10**(-6), epoch=100):
        error = 1
        iter_num = 0

        U = U0
        while error > tol and iter_num < epoch:
            b = self.hjb_sys_vec(U, M, idx)
            jacobi = jacfwd(self.hjb_sys_vec)(U, M, idx)
            dz = jnp.linalg.solve(jacobi, b.flatten())
            U = U - lr * dz

            error = jnp.dot(dz, dz)
            print("             the error of solving hjb is {}".format(error))

            iter_num = iter_num + 1

        return U

    def fp(self, Uk, Mk1, Mk, idx):
        Dt_M = (Mk1 - Mk) / self.dt

        Mk1R = jnp.roll(Mk1, -1)
        Mk1L = jnp.roll(Mk1, 1)

        Delta_M = - (2 * Mk1 - Mk1R - Mk1L) / self.h**2

        adj = self.fp_linearized_part(Uk, Mk1, idx)

        return Dt_M - self.nu * Delta_M + adj

    def solve_fp(self, M0, U1, U2):
        jacobi = jacfwd(self.fp_sys_vec)(M0, U1, U2)
        b = self.fp_sys_vec(jnp.zeros(len(jacobi)), U1, U2)
        M = jnp.linalg.lstsq(jacobi, -b)
        return M[0]

    def hjb_sys(self, U, M, idx):
        ts = jnp.linspace(0, self.T, self.Nt + 1)
        hjbs = vmap(self.hjb, in_axes=(0, 0, 0, 0, None))(ts[:-1], U[1:, :], U[:-1, :], M[1:, :], idx)
        return hjbs

    def fp_sys(self, U, M, idx):
        fps = vmap(self.fp, in_axes=(0, 0, 0, None))(U[:-1, :], M[1:, :], M[:-1, :], idx)

        return fps

    def hjb_sys_vec(self, Uvec, Mvec, idx):
        U, M = self.prolong(Uvec, Mvec, idx)
        return self.hjb_sys(U, M, idx).flatten()

    def fp_sys_vec(self, Mvec, Uvec, idx):
        U, M = self.prolong(Uvec, Mvec, idx)

        return self.fp_sys(U, M, idx).flatten()

    def solve(self, tol=10**(-6), epoch=100, hjb_lr=1, hjb_epoch=100):
        U1 = jnp.zeros((self.Nt, self.N)).flatten()
        U2 = jnp.zeros((self.Nt, self.N)).flatten()

        M1 = jnp.zeros((self.Nt, self.N)).flatten()
        M2 = jnp.zeros((self.Nt, self.N)).flatten()

        error = 1
        iter_num = 0

        while error > tol and iter_num < epoch:
            new_U1 = self.solve_hjb(U1, M1, hjb_lr, epoch=hjb_epoch, idx=0)
            new_U2 = self.solve_hjb(U2, M2, hjb_lr, epoch=hjb_epoch, idx=1)

            new_M1 = self.solve_fp(M1, new_U1, 0)
            new_M2 = self.solve_fp(M2, new_U2, 1)

            U_err = jnp.dot(new_U1 - U1, new_U1 - U1) + jnp.dot(new_U2 - U2, new_U2 - U2)
            M_err = jnp.dot(new_M1 - M1, new_M1 - M1) + jnp.dot(new_M2 - M2, new_M2 - M2)
            error = U_err + M_err

            print('MFG error: {}'.format(error))

            U1, U2, M1, M2 = new_U1, new_U2, new_M1, new_M2
            iter_num += 1

        U1, M1 = self.prolong(U1, M1, 0)
        U2, M2 = self.prolong(U2, M2, 1)

        return U1, M1, U2, M2


np.set_printoptions(precision=20)

cfg = munch.munchify({
    'T' : 1,
    'Nt': 60,
    'xl': -6,
    'xr': 6,
    'N' : 50,
    'nu': 1,
    'alphas': [1, 1, 1],
    'sigmas': [2, 1, 1],
    'lambdas':[0.6, 0.2, 0.2],
    'eps': 0.5,
    'hjb_epoch': 200,
    'hjb_lr': 1,
    'epoch': 150,
    'lr': 0.7,
    'tol' : 10 ** (-6),
})


solver = TwoPopSimplified(T=cfg.T, Nt=cfg.Nt, xl=cfg.xl, xr=cfg.xr, N=cfg.N, nu=cfg.nu, alphas=cfg.alphas, sigmas=cfg.sigmas, lambdas=cfg.lambdas, eps=cfg.eps)

TT = jnp.linspace(0, cfg.T, cfg.Nt + 1)
XX = jnp.linspace(cfg.xl, cfg.xr, cfg.N, endpoint=False)
TT, XX = jnp.meshgrid(TT, XX)

U1, M1, U2, M2 = solver.solve(cfg.tol, cfg.epoch, cfg.hjb_lr, cfg.hjb_epoch)

TT = np.linspace(0, cfg.T, cfg.Nt + 1)
XX = np.linspace(-10, 10, cfg.N, endpoint=False)

# Final value function for both populations
plt.figure()
plt.plot(XX, U1[-1, :], label='U1(T)')
plt.plot(XX, U2[-1, :], label='U2(T)')
plt.title('Final Value Function')
plt.xlabel('x')
plt.ylabel('u(T)')
plt.legend()
plt.show()

# Final distribution for both populations
plt.figure()
plt.plot(XX, M1[-1, :], label="Pop. 1")
plt.plot(XX, M2[-1, :], label="Pop. 2")
plt.axvline(x=x_d1, color='Blue', linestyle='--', linewidth=1, label=r'Line at $x_{d1}=$' + f'{x_d1}')
plt.axvline(x=x_d2, color='Green', linestyle='--', linewidth=1, label=r'Line at $x_{d2}=$' + f'{x_d3}')
plt.xlabel('x')
plt.ylabel('m(T)')
plt.legend()
plt.savefig("final_good.png", dpi=300)
plt.show()

# 3D surface plot for value function U1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(XX, TT)
ax.plot_surface(X, Y, U1, cmap='viridis')
ax.set_title('U1 over Time and Space')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U1')
plt.show()

# 3D surface plot for value function U2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U2, cmap='viridis')
ax.set_title('U2 over Time and Space')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('U2')
plt.show()


# 3D surface plot for distribution M1
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, M1, cmap='viridis')
ax.set_title('M1 over Time and Space')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('M1')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, M2, cmap='viridis')
ax.set_title('M2 over Time and Space')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('M2')
plt.show()
